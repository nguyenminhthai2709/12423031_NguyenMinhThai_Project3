"""Kiểm thử lõi: quy tắc chia G1 (giao thức B của TaxPro-CL) trên dữ liệu đồ chơi, và split thật không rò rỉ."""

import json
from pathlib import Path

import pytest

from config_path.config_path import DATASET_NAME  # noqa: E402
from tools.data.build_g1 import build_split, deduplicate_latest, iterative_k_core, restrict_to_catalog, split_counts

ROOT = Path(__file__).resolve().parents[1]


def toy_records():
    """6 user × 8 phim (phim chia hết cho 4 bị chấm 3 sao); user 7 có 1 tương tác; phim 99 chỉ có 2 user."""
    records = []
    for user in range(1, 7):
        for item in range(1, 9):
            records.append((user, item, 3 if item % 4 == 0 else 5, 1000 + item))
    records.append((7, 1, 5, 1))                       # user 7 chỉ 1 tương tác dương → bị 5-core loại
    records += [(1, 99, 5, 1), (2, 99, 5, 1)]          # phim 99 chỉ 2 tương tác → bị 5-core loại
    records.append((1, 1, 2, 5000))                    # bản ghi lặp mới hơn của (1, 1): rating 2 → không còn dương
    return records


def test_split_ratios():
    """round(0,7n) train, round(0,1n) validation (làm tròn ngân hàng như TaxPro-CL), còn lại test"""
    assert split_counts(5) == (4, 0, 1)
    assert split_counts(10) == (7, 1, 2)
    assert split_counts(23) == (16, 2, 5)
    assert all(sum(split_counts(n)) == n for n in range(1, 60))


def test_deduplicate_keeps_latest():
    """Khử trùng cặp user–phim, giữ bản ghi muộn nhất"""
    kept, removed = deduplicate_latest(toy_records())
    assert removed == 1
    assert [r for r in kept if r[0] == 1 and r[1] == 1][0][2] == 2


def test_iterative_kcore_users_and_items():
    """5-core lặp: mọi user và mọi phim còn lại có >= 5 tương tác"""
    pairs = {(u, i) for u in range(5) for i in range(5)} | {(9, 0), (0, 9)}
    core, stats = iterative_k_core(pairs, 5)
    assert core == {(u, i) for u in range(5) for i in range(5)} and stats["users"] == 5 and stats["items"] == 5


def test_split_rules_and_determinism():
    """Rating >= 4, 5-core user và phim, ba tập rời nhau, cùng seed cho cùng kết quả"""
    train, validation, test, stats = build_split(toy_records())
    assert 7 not in train and stats["users_removed_min_positives"] == 1
    assert all(99 not in train[u] + validation[u] + test[u] for u in train)
    for user in train:
        assert not (set(train[user]) & set(validation[user])) and not (set(train[user]) & set(test[user]))
        assert not (set(validation[user]) & set(test[user]))
        assert (len(train[user]), len(validation[user]), len(test[user])) == split_counts(
            len(train[user]) + len(validation[user]) + len(test[user]))
    assert 1 not in train[1] + validation[1] + test[1]
    again = build_split(toy_records())
    assert (train, validation, test) == again[:3]


def test_catalog_only_train_items():
    """Danh mục chỉ gồm phim có tương tác train; đích ngoài danh mục bị tách và đếm"""
    catalog, removed = restrict_to_catalog({1: [1, 2]}, {1: [3]}, {1: [2, 4]})
    assert catalog == [1, 2]
    assert removed["validation"]["pairs"] == 1 and removed["test"]["pairs"] == 1


def test_prepared_split_is_leak_free():
    """Split G1 đã dựng: train/validation/test rời nhau và mọi đích nằm trong danh mục train"""
    topic = ROOT / "topic.json"
    if topic.is_file() and "ml-100k" not in json.loads(topic.read_text(encoding="utf-8")).get("datasets", ["ml-100k"]):
        pytest.skip("KHÔNG ÁP DỤNG: đề tài không dùng split G1 của MovieLens")
    directory = ROOT / "dataset_verify" / DATASET_NAME
    if not (directory / "split_manifest.json").is_file():
        pytest.skip("CHƯA LÀM: chưa chạy python -m tools.data.prepare")
    parts = {}
    for name in ("train", "validation", "test"):
        pairs = set()
        for line in (directory / "{}.txt".format(name)).read_text(encoding="utf-8").splitlines():
            values = [int(v) for v in line.split()]
            pairs.update((values[0], item) for item in values[1:])
        parts[name] = pairs
    assert not (parts["train"] & parts["validation"]) and not (parts["train"] & parts["test"])
    assert not (parts["validation"] & parts["test"])
    catalog = {item for _, item in parts["train"]}
    assert all(item in catalog for _, item in parts["validation"] | parts["test"])
    manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_seed"] == 42 and manifest["positive_threshold"] == 4
    assert manifest.get("item_kcore_filter") is True, "split cũ (chưa lọc 5-core): chạy lại python -m tools.data.prepare"
