"""Nạp split đã dựng (train/validation/test.txt, mỗi dòng: user item1 item2 ...) và lấy mẫu âm.

Kế thừa từ ID-GRec. Quy ước: mã user và mã phim đã được ánh xạ liên tiếp từ 0 bởi tools/data/build_g1.py.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from config_path.config_path import dataset_split_file, listed_item_count


def _sample_negative_items(users, catalog, user_item_net, seed):
    """Lấy mẫu âm đều trên DANH MỤC TRAIN (phim có >= 1 tương tác train), loại các phim dương của chính user.

    Bộ đề (G1): "Lấy mẫu âm đều từ danh mục train, loại các dương train của người dùng." Với split G1 mọi
    phim đều thuộc danh mục train; với split C1 (cold-start) phim cold không bao giờ được lấy làm mẫu âm.
    Cấu hình `negative_catalog = all` lấy mẫu trên toàn bộ danh mục (dùng khi cần tái lập một mô hình nền
    đã huấn luyện theo cách đó, ví dụ Amazon-Book của đề tài NCKH).
    """
    if len(users) == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(np.diff(user_item_net.indptr)[users] >= len(catalog)):
        raise ValueError("Có user đã tương tác với mọi phim, không lấy được mẫu âm")
    rng = np.random.RandomState(seed)
    negatives = catalog[rng.randint(0, len(catalog), size=len(users))]
    positive = np.asarray(user_item_net[users, negatives]).reshape(-1) > 0
    while np.any(positive):
        negatives[positive] = catalog[rng.randint(0, len(catalog), size=int(positive.sum()))]
        positive = np.asarray(user_item_net[users, negatives]).reshape(-1) > 0
    return negatives


class Data(object):
    def __init__(self, path, config, logger=None):
        self.path = str(path)
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.num_users = 0
        self.num_items = 0
        self.current_epoch = 0
        self.training_output_dir = None
        self.validation_history = []
        self.num_worker = max(1, int(config.get("num_worker", 4)))
        self.load_data()

    def load_data(self):
        paths = {split: dataset_split_file(self.path, split) for split in ("train", "validation", "test")}
        for split, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    "Thiếu {} ({}). Chạy trước: python -m tools.data.prepare".format(split, path))
        self.train_user, self.train_item, self.num_train = self.read_ratings(paths["train"])
        self.validation_user, self.validation_item, self.num_validation = self.read_ratings(paths["validation"])
        self.test_user, self.test_item, self.num_test = self.read_ratings(paths["test"])
        self.num_users += 1
        self.num_items = max(self.num_items + 1, listed_item_count(self.path))
        self.num_nodes = self.num_users + self.num_items
        self.user_item_net = sp.csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
                                           shape=(self.num_users, self.num_items))
        validation_net = sp.csr_matrix((np.ones(len(self.validation_user)), (self.validation_user, self.validation_item)),
                                       shape=(self.num_users, self.num_items))
        self.train_validation_net = (self.user_item_net + validation_net).tocsr()
        self.train_catalog = np.flatnonzero(self.item_train_counts() > 0).astype(np.int64)
        policy = str(self.config.get("negative_catalog", "train"))
        if policy not in ("train", "all"):
            raise ValueError("negative_catalog phải là train hoặc all")
        self.negative_catalog = (self.train_catalog if policy == "train"
                                 else np.arange(self.num_items, dtype=np.int64))
        self.split_exclusions = self.read_split_exclusions()
        self.validation_dict = self.build_interaction_dict(self.validation_user, self.validation_item)
        self.test_dict = self.build_interaction_dict(self.test_user, self.test_item)
        self.logger.info("Dữ liệu: users=%d items=%d train=%d validation=%d test=%d", self.num_users, self.num_items,
                         self.num_train, self.num_validation, self.num_test)

    def read_ratings(self, file_name):
        users, items, total = [], [], 0
        with open(file_name, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    values = [int(value) for value in text.split()]
                except ValueError as error:
                    raise ValueError("Dòng không hợp lệ {}:{}".format(file_name, line_number)) from error
                user, positives = values[0], values[1:]
                self.num_users = max(self.num_users, user)
                if not positives:
                    continue
                self.num_items = max(self.num_items, max(positives))
                users.extend([user] * len(positives))
                items.extend(positives)
                total += len(positives)
        return np.array(users, dtype=np.int64), np.array(items, dtype=np.int64), total

    def item_train_counts(self):
        """Số tương tác train của từng phim (mức hỗ trợ dùng để chia nhóm 1–5, 1–10, > 10)."""
        return np.asarray(self.user_item_net.sum(axis=0)).reshape(-1).astype(np.int64)

    def user_train_counts(self):
        return np.asarray(self.user_item_net.sum(axis=1)).reshape(-1).astype(np.int64)

    def sample_data_to_train_all(self):
        """Mỗi cặp dương trong train đi kèm một phim âm. Trả về mảng [n, 3] = (user, pos, neg)."""
        num_samples = len(self.train_user)
        workers = min(self.num_worker, max(num_samples, 1))
        chunks = np.array_split(np.arange(num_samples), workers)
        seeds = [int(np.random.randint(0, 2 ** 31 - 1)) for _ in chunks]
        if workers == 1:
            negatives = [_sample_negative_items(self.train_user[chunks[0]], self.negative_catalog, self.user_item_net,
                                                seeds[0])]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_sample_negative_items, self.train_user[idx], self.negative_catalog,
                                           self.user_item_net, seed) for idx, seed in zip(chunks, seeds)]
                negatives = [future.result() for future in futures]
        return np.column_stack((self.train_user, self.train_item, np.concatenate(negatives)))

    def read_split_exclusions(self):
        """Phim bị loại khỏi ứng viên của cả một split (chỉ split C1 có tệp candidate_exclusions.json).

        C1: validation không xếp hạng phim cold-test; test không xếp hạng phim cold-validation.
        """
        path = Path(self.path) / "candidate_exclusions.json"
        if not path.is_file():
            return {"validation": np.empty(0, dtype=np.int64), "test": np.empty(0, dtype=np.int64)}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {split: np.asarray(payload.get(split, []), dtype=np.int64) for split in ("validation", "test")}

    def get_user_pos_items_for_evaluation(self, users, split):
        """Phim cần loại khỏi danh sách ứng viên: validation loại dương train; test loại thêm dương validation.

        Với split C1, mỗi user còn bị loại thêm các phim cold không thuộc split đang đánh giá.
        """
        matrix = self.user_item_net if split == "validation" else self.train_validation_net
        extra = self.split_exclusions[split]
        if len(extra) == 0:
            return [matrix[user].nonzero()[1] for user in users]
        return [np.union1d(matrix[user].nonzero()[1], extra) for user in users]

    @staticmethod
    def build_interaction_dict(users, items):
        interactions = {}
        for user, item in zip(users, items):
            interactions.setdefault(int(user), []).append(int(item))
        return interactions
