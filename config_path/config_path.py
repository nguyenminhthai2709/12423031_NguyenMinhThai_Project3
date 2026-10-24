"""Đường dẫn chuẩn của dự án. Mọi script đều lấy đường dẫn từ đây, không tự ghép chuỗi."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURE_DIR = PROJECT_ROOT / "configure"            # cấu hình mặc định của từng mô hình / đề tài
DATASET_DIR = PROJECT_ROOT / "dataset"                # dữ liệu gốc tải về (KHÔNG chia sẻ lại cho người khác)
DATASET_VERIFY_DIR = PROJECT_ROOT / "dataset_verify"  # split đã dựng, mô hình chỉ đọc từ đây
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"      # nhóm item và tập đích theo nhóm (giao thức đánh giá)
METADATA_DIR = PROJECT_ROOT / "metadata"              # thể loại phim, cây thể loại quy ước
LOG_DIR = Path(os.environ.get("P3_LOG_ROOT") or (PROJECT_ROOT / "log")).resolve()  # run, checkpoint
# MovieLens dùng cho 20 đề tài. Chọn bằng biến môi trường P3_DATASET (mặc định ml-100k), ví dụ trên PowerShell:
#   $env:P3_DATASET = "ml-1m"; python -m tools.data.prepare
# MD5 khớp tệp <tên>.zip.md5 do GroupLens công bố; SHA-256 tính trên đúng tệp đó.
MOVIELENS = {
    "ml-100k": {"url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
                "md5": "0e33842e24a9c977be4e0107933c0723",
                "sha256": "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"},
    "ml-1m": {"url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
              "md5": "c4d9eecfca2ab87c1945afe126590906",
              "sha256": "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"},
}
DATASET_NAME = os.environ.get("P3_DATASET", "ml-100k").strip() or "ml-100k"
if DATASET_NAME not in MOVIELENS:
    raise ValueError("P3_DATASET = {!r} không hợp lệ; chọn một trong: {}".format(DATASET_NAME, ", ".join(MOVIELENS)))
ML100K_URL = MOVIELENS["ml-100k"]["url"]
ML100K_MD5 = MOVIELENS["ml-100k"]["md5"]
ML100K_SHA256 = MOVIELENS["ml-100k"]["sha256"]

# Kết quả ML-100K giữ ở results/ như cũ; dataset khác ghi riêng vào results/<dataset>/ để không ghi đè.
RESULT_DIR = PROJECT_ROOT / "results" if DATASET_NAME == "ml-100k" else PROJECT_ROOT / "results" / DATASET_NAME


def resolve_dataset(name):
    """Tên dataset trong cấu hình viết cho ML-100K ("ml-100k", "ml-100k-c1") → dataset MovieLens đang chọn.

    Dataset khác (ví dụ amazon-book của NCKH) giữ nguyên.
    """
    name = str(name)
    if name == "ml-100k" or name.startswith("ml-100k-"):
        return DATASET_NAME + name[len("ml-100k"):]
    return name

TRAIN_FILE_NAME = "train.txt"
VALIDATION_FILE_NAME = "validation.txt"
TEST_FILE_NAME = "test.txt"
ITEM_GENRES_FILE_NAME = "item_genres.csv"
GENRE_NAMES_FILE_NAME = "genre_names.json"
BEST_VALIDATION_MODEL_FILE_NAME = "best_validation_model.pt"
LAST_MODEL_FILE_NAME = "last_model.pt"
VALIDATION_METRICS_FILE_NAME = "validation_metrics.json"
FINAL_TEST_METRICS_FILE_NAME = "final_test_metrics.json"
FINAL_TEST_GROUP_CSV_FILE_NAME = "final_test_groups.csv"
RUN_MANIFEST_FILE_NAME = "run_manifest.json"


def model_config_file(model_name):
    return CONFIGURE_DIR / (str(model_name) + ".txt")


def raw_dataset_dir():
    """Thư mục dữ liệu gốc đã giải nén; ML-1M có thêm u.data/u.item/u.genre chuẩn hóa (xem download_ml100k)."""
    return DATASET_DIR / DATASET_NAME


def verified_dataset_dir(dataset_name=DATASET_NAME):
    return DATASET_VERIFY_DIR / resolve_dataset(dataset_name)


def dataset_split_file(dataset_directory, split):
    names = {"train": TRAIN_FILE_NAME, "validation": VALIDATION_FILE_NAME, "test": TEST_FILE_NAME}
    return Path(dataset_directory) / names[split]


def listed_item_count(dataset_directory):
    """Số phim ghi trong item_list.txt (bỏ dòng tiêu đề); 0 nếu không có tệp.

    Split C1 có thể có phim cold không còn xuất hiện trong tệp split nào; số phim phải lấy theo danh mục.
    """
    path = Path(dataset_directory) / "item_list.txt"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return max(0, sum(1 for line in stream if line.strip()) - 1)


def protocol_dir(dataset_name=DATASET_NAME):
    return PREPROCESSED_DIR / "evaluation_protocol" / resolve_dataset(dataset_name)


def item_genres_file(dataset_name=DATASET_NAME):
    return METADATA_DIR / resolve_dataset(dataset_name) / ITEM_GENRES_FILE_NAME


def genre_names_file(dataset_name=DATASET_NAME):
    return METADATA_DIR / resolve_dataset(dataset_name) / GENRE_NAMES_FILE_NAME


def run_dir(model_name, dataset_name, config_id, seed):
    """log/<mô hình>/<dataset>/<mã cấu hình>/seed<k>/ : một run = một seed."""
    return LOG_DIR / str(model_name) / resolve_dataset(dataset_name) / str(config_id) / ("seed" + str(seed))


def checkpoint_file(output_dir, kind):
    names = {"best_validation": BEST_VALIDATION_MODEL_FILE_NAME, "last": LAST_MODEL_FILE_NAME}
    return Path(output_dir) / names[kind]


def result_file(output_dir, kind):
    names = {"validation": VALIDATION_METRICS_FILE_NAME, "final_test": FINAL_TEST_METRICS_FILE_NAME,
             "final_test_groups": FINAL_TEST_GROUP_CSV_FILE_NAME, "manifest": RUN_MANIFEST_FILE_NAME}
    return Path(output_dir) / names[kind]


def temporary_file(target):
    target = Path(target)
    return target.with_name(target.name + ".tmp")


def relative_to_project(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
