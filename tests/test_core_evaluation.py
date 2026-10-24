"""Kiểm thử lõi: chỉ số, theo nhóm, mặt nạ ứng viên, phá hòa cố định."""

import numpy as np
import pytest
import torch

from utility.utility_train.batch_test import rank_top_k, tie_break_permutation
from utility.utility_train.group_evaluator import evaluate_rankings, is_exploratory, user_ndcg, user_recall


def test_page9_hand_example():
    """Ví dụ tự kiểm tra trang 9: Recall@3 = 1, NDCG@3 ≈ 0,6934, NDCG nhóm ≈ 0,6309"""
    X, A, B = 0, 1, 2
    ranking = [X, A, B]
    assert user_recall(ranking, [A, B], 3) == 1.0
    assert user_ndcg(ranking, [A, B], 3) == pytest.approx(0.6934, abs=1e-4)
    result = evaluate_rankings({7: ranking}, {"overall": {7: [A, B]}, "long_tail_1_10": {7: [A]}}, ks=(3,))
    assert result["long_tail_1_10"]["recall"]["3"] == 1.0
    assert result["long_tail_1_10"]["ndcg"]["3"] == pytest.approx(0.6309, abs=1e-4)


def test_group_excludes_users_without_targets():
    """User không có đích trong nhóm không tham gia trung bình của nhóm"""
    result = evaluate_rankings({1: [5, 6], 2: [7, 8]}, {"overall": {1: [5], 2: [9]}, "g": {1: [5]}}, ks=(2,))
    assert result["g"]["support"]["users"] == 1 and result["g"]["recall"]["2"] == 1.0
    assert result["overall"]["recall"]["2"] == 0.5


def test_exploratory_flag():
    """Nhóm dưới 30 user hoặc dưới 100 đích được gắn nhãn thăm dò"""
    assert is_exploratory({"users": 29, "targets": 500, "items": 10})
    assert is_exploratory({"users": 100, "targets": 99, "items": 10})
    assert not is_exploratory({"users": 30, "targets": 100, "items": 10})


def test_masked_items_never_ranked():
    """Phim đã thấy (bị loại) không bao giờ lọt vào top-K, kể cả khi điểm cao nhất"""
    scores = torch.tensor([[9.0, 1.0, 2.0, 3.0]])
    top = rank_top_k(scores, [np.array([0])], 3, tie_break_permutation(4, 1))
    assert 0 not in top[0].tolist() and top[0].tolist() == [3, 2, 1]


def test_tie_break_is_fixed_permutation():
    """Khi bằng điểm, thứ tự theo hoán vị cố định, giống nhau giữa các lần chạy"""
    permutation = tie_break_permutation(6, 12345)
    scores = torch.zeros((1, 6))
    first = rank_top_k(scores, [np.array([], dtype=int)], 6, permutation)[0].tolist()
    assert first == permutation.tolist()
    assert rank_top_k(scores, [np.array([], dtype=int)], 6, permutation)[0].tolist() == first


def _toy_split(directory, exclusions=None):
    (directory / "train.txt").write_text("0 0 1\n1 1 2\n", encoding="utf-8")
    (directory / "validation.txt").write_text("0 3\n1 0\n", encoding="utf-8")
    (directory / "test.txt").write_text("0 4\n1 3\n", encoding="utf-8")
    if exclusions is not None:
        import json
        (directory / "candidate_exclusions.json").write_text(json.dumps(exclusions), encoding="utf-8")


def test_split_exclusions_and_train_catalog_policy(tmp_path):
    """Split C1: validation không xếp hạng cold-test, test không xếp hạng cold-validation; train_catalog bỏ phim cold"""
    from utility.utility_data.data_loader import Data
    from utility.utility_train.batch_test import compute_rankings
    _toy_split(tmp_path, {"validation": [4], "test": [3]})
    data = Data(str(tmp_path), {"num_worker": 1})
    assert list(data.train_catalog) == [0, 1, 2]
    excluded = data.get_user_pos_items_for_evaluation(np.array([0]), "validation")[0]
    assert set(excluded) == {0, 1, 4}
    excluded = data.get_user_pos_items_for_evaluation(np.array([0]), "test")[0]
    assert set(excluded) == {0, 1, 3}

    def score(users):
        return torch.tensor([[0.0, 0.1, 0.2, 0.9, 0.8]] * len(users))
    assert compute_rankings(data, score, "validation", 2, 8, 1)[0] == [3, 2]
    assert compute_rankings(data, score, "validation", 2, 8, 1, "train_catalog")[0][0] == 2


def test_split_without_exclusion_file_is_unchanged(tmp_path):
    """Split G1 (không có candidate_exclusions.json): chỉ loại dương train/validation như cũ"""
    from utility.utility_data.data_loader import Data
    _toy_split(tmp_path)
    data = Data(str(tmp_path), {"num_worker": 1})
    assert set(data.get_user_pos_items_for_evaluation(np.array([1]), "test")[0]) == {0, 1, 2}


def test_selection_group():
    """Chọn checkpoint theo selection_group; mặc định là overall"""
    from utility.utility_train.batch_test import _selection
    result = {"overall": {"ndcg": {"10": 0.1}}, "warm_1plus": {"ndcg": {"10": 0.3}}}
    assert _selection({}, result) == ("ndcg@10", 0.1)
    assert _selection({"selection_group": "warm_1plus"}, result) == ("ndcg@10[warm_1plus]", 0.3)
    with pytest.raises(ValueError):
        _selection({"selection_group": "cold_0"}, result)


def test_negative_catalog_policy(tmp_path):
    """Mẫu âm: mặc định chỉ lấy từ danh mục train; negative_catalog = all lấy từ toàn bộ danh mục"""
    from utility.utility_data.data_loader import Data
    _toy_split(tmp_path)
    assert list(Data(str(tmp_path), {"num_worker": 1}).negative_catalog) == [0, 1, 2]
    data = Data(str(tmp_path), {"num_worker": 1, "negative_catalog": "all"})
    assert list(data.negative_catalog) == [0, 1, 2, 3, 4]
    triples = data.sample_data_to_train_all()
    assert all(data.user_item_net[u, n] == 0 for u, _, n in triples)
