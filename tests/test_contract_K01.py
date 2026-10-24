"""Test hợp đồng K01 – mục 8 "Tiêu chí hoàn thành" (phương án 1 SV, cây MovieLens quy ước).

(1) Cây đối chiếu được với nguồn (phụ lục) và quy tắc xác định cha: mỗi lá thuộc đúng một cha.
(2) Prototype cha khớp phép tính trực tiếp từ phim con.
(3) Ghi số phim mất metadata và không coi gốc duy nhất là tín hiệu phân cấp.
"""

import json

import numpy as np

from models import K01_two_level as K
from tests.contract_helpers import ROOT, call_or_pending, require_csv
from utility.utility_function import prototype as P

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def test_tc1_tree_matches_appendix():
    """TC1: 18 lá MovieLens (bỏ unknown), 5 cha, mỗi lá thuộc đúng một cha; tên lá đúng chuỗi trong u.genre"""
    tree = json.loads((ROOT / "metadata" / "convention_tree.json").read_text(encoding="utf-8"))
    leaves = [leaf for parent in tree["parents"] for leaf in parent["leaves"]]
    assert sorted(leaves) == sorted(GENRES) and len(leaves) == len(set(leaves)) == 18
    assert len(tree["parents"]) == 5
    parent_of, parent_ids = K.load_tree(ROOT / "metadata" / "convention_tree.json")
    labels = np.zeros((1, 18), dtype=bool)
    membership = call_or_pending(K.parent_membership, labels, GENRES, parent_of, parent_ids)
    assert membership.shape == (1, 5)


def test_tc2_parent_prototype_is_direct_mean_of_children():
    """TC2: phim có hai lá cùng cha chỉ góp một lần; prototype cha = trung bình trực tiếp phim con phân biệt"""
    names = ["A", "B", "C"]
    parent_of, parent_ids = {"A": "p", "B": "p", "C": "q"}, ["p", "q"]
    labels = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    membership = call_or_pending(K.parent_membership, labels, names, parent_of, parent_ids)
    assert membership.tolist() == [[True, False], [True, False], [True, False], [False, True]]
    items = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 3.0], [1.0, 1.0]])
    counts = np.array([1, 1, 1, 1])
    prototypes, support = P.label_prototypes(items, membership, counts)
    assert list(support) == [3, 1] and np.allclose(prototypes[0], items[:3].mean(axis=0))
    leaf_centres, _ = P.label_prototypes(items, labels, counts)
    assert not np.allclose(prototypes[0], leaf_centres[:2].mean(axis=0)), "không lấy trung bình đều các tâm lá"


def test_tc2b_two_level_vectors():
    """TC2b: vector hỗn hợp = cân thang G2 của 0,5 lá + 0,5 cha; phim không nhãn quay về CF"""
    labels = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    membership = np.array([[1, 0], [1, 0], [0, 0]], dtype=bool)
    items = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    counts = np.array([1, 1, 1])
    mix, valid = call_or_pending(K.two_level_vectors, items, labels, membership, counts, "mix")
    assert list(valid) == [True, True, False]
    expected, _ = P.scale_like_items(np.array([[1.5, 0.5], [0.5, 1.5], [0.0, 0.0]]), items, counts)
    assert np.allclose(mix, expected)


def test_tc3_coverage_and_no_single_root():
    """TC3: có bảng độ phủ (số phim mất nhãn, số node mỗi cấp), không dùng gốc duy nhất; có kết quả 2 seed"""
    coverage = require_csv("results/K01_tree_coverage.csv", ["items_without_leaf", "leaf_nodes", "parent_nodes",
                                                             "root_used"])
    assert int(coverage[0]["parent_nodes"]) >= 2 and coverage[0]["root_used"] == "0"
    rows = require_csv("results/K01_test.csv", ["seed", "method", "group", "ndcg@10"])
    assert {"CF", "leaf", "mix"} <= {r["method"] for r in rows} and len({r["seed"] for r in rows}) >= 2
