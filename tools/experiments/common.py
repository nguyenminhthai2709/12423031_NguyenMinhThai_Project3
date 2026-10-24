"""Hàm dùng chung cho script thí nghiệm của đề tài (tools/experiments/run_<MÃ>.py)."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import (CONFIGURE_DIR, DATASET_NAME, RESULT_DIR, relative_to_project,  # noqa: E402
                                     resolve_dataset, verified_dataset_dir)
from utility.utility_data.data_loader import Data  # noqa: E402
from utility.utility_data.genres import load_item_genres  # noqa: E402
from utility.utility_function.tools import read_configuration  # noqa: E402
from utility.utility_train import batch_test, posthoc  # noqa: E402
from utility.utility_train.group_evaluator import evaluate_rankings, load_protocol, result_rows  # noqa: E402


def topic_config(code):
    """Đọc configure/<MÃ>.txt; các khóa dạng danh sách viết như JSON, ví dụ seeds = [42, 0, 1]."""
    config = read_configuration(str(CONFIGURE_DIR / (code + ".txt")), code)
    if "dataset" in config:
        config["dataset"] = resolve_dataset(config["dataset"])
    return config


def json_value(config, key, default=None):
    if key not in config:
        return default
    return json.loads(config[key])


def load_g1(config=None, dataset_name=DATASET_NAME):
    config = dict(config or {})
    config["dataset"] = resolve_dataset(config.get("dataset", dataset_name))
    return Data(str(verified_dataset_dir(config["dataset"])), config)


def load_base(model_name, seed, tag="base", dataset_name=DATASET_NAME, num_worker=None):
    """Nạp checkpoint mô hình nền đã huấn luyện bằng main.py (--tag base).

    num_worker: số luồng lấy mẫu âm khi đề tài học thêm trên checkpoint (ví dụ học cổng); mặc định giữ giá trị của run.
    """
    bundle = posthoc.load_run(posthoc.latest_run(model_name, seed, dataset_name, tag=tag))
    if num_worker is not None:
        bundle.dataset.num_worker = max(1, int(num_worker))
    return bundle


def item_labels(dataset, dataset_name=DATASET_NAME):
    labels, names = load_item_genres(dataset_name, num_items=dataset.num_items)
    return labels, names


def evaluate(dataset, config, score_fn, split, keep_per_user=False, extra_groups=None):
    """Đánh giá score_fn trên split; extra_groups: {tên: hàm(item_id) -> bool} để cắt thêm nhóm theo phim."""
    ks = batch_test.parse_ks(config)
    rankings = batch_test.compute_rankings(dataset, score_fn, split, max(ks), int(config.get("test_batch_size", 1024)),
                                           int(config.get("tiebreak_seed", 12345)),
                                           config.get("candidate_policy", "all"))
    definitions, targets = load_protocol(batch_test.protocol_directory(config), split)
    for name, predicate in (extra_groups or {}).items():
        targets[name] = {u: [i for i in items if predicate(i)] for u, items in targets["overall"].items()}
        definitions[name] = {"label": name}
    return evaluate_rankings(rankings, targets, ks, keep_per_user), definitions


def rows(result, definitions, config, **extra):
    """Dòng CSV cho từng nhóm, kèm các cột mô tả (method, seed, split, ...)."""
    output = []
    for row in result_rows(result, batch_test.parse_ks(config), definitions):
        merged = dict(extra)
        merged.update(row)
        output.append(merged)
    return output


def write_csv(name, records):
    """Ghi results/<name> (CSV, UTF-8). Trả về đường dẫn."""
    if not records:
        raise ValueError("Không có dòng nào để ghi vào {}".format(name))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / name
    columns = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    print("Đã ghi", relative_to_project(path), "({} dòng)".format(len(records)))
    return path


def matrix_scores(scores):
    """score_fn cho evaluate() từ ma trận điểm có sẵn [users x items]."""
    scores = np.asarray(scores, dtype=np.float64)

    def score(batch):
        return scores[np.asarray(batch)]
    return score


def dot_scores(user_vectors, item_vectors):
    """score_fn cho evaluate(): điểm = <u, v_i>."""
    users = np.asarray(user_vectors, dtype=np.float64)
    items = np.asarray(item_vectors, dtype=np.float64)

    def score(batch):
        return users[np.asarray(batch)] @ items.T
    return score
