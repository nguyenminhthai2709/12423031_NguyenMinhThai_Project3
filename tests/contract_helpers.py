"""Hàm hỗ trợ cho test hợp đồng của đề tài.

- Hàm còn để TODO (raise NotImplementedError) → test hiện "CHƯA LÀM" (skip), không phải lỗi.
- Tệp kết quả chưa có → "CHƯA LÀM". Tệp có nhưng sai cột → "SAI".
"""

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import RESULT_DIR, resolve_dataset  # noqa: E402


def project_path(relative_path):
    """Đường dẫn viết cho ML-100K → dataset đang chọn: results/ → RESULT_DIR; thành phần ml-100k* → P3_DATASET."""
    parts = Path(relative_path).parts
    if parts and parts[0] == "results":
        return RESULT_DIR.joinpath(*parts[1:])
    return ROOT.joinpath(*[resolve_dataset(part) for part in parts])


def call_or_pending(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except NotImplementedError as error:
        pytest.skip("CHƯA LÀM: {}".format(error or function.__name__))


def require_csv(relative_path, required_columns):
    path = project_path(relative_path)
    if not path.is_file():
        pytest.skip("CHƯA CÓ KẾT QUẢ: {}".format(relative_path))
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows, "{} không có dòng dữ liệu".format(relative_path)
    missing = [column for column in required_columns if column not in rows[0]]
    assert not missing, "{} thiếu cột: {}".format(relative_path, missing)
    return rows


def require_file(relative_path):
    path = project_path(relative_path)
    if not path.is_file():
        pytest.skip("CHƯA CÓ: {}".format(relative_path))
    return path
