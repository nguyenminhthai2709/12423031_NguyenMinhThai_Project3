"""Kiểm thử lõi: prototype G2 và bộ trộn (bộ đề, trang 10: "Bộ kiểm tra cần gồm...")."""

import numpy as np

from utility.utility_function import prototype as P


def toy():
    vectors = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0], [1.0, 1.0]])
    labels = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)  # phim 3 không có nhãn
    counts = np.array([5, 1, 2, 3])
    return vectors, labels, counts


def test_prototype_is_mean_of_train_items():
    """Prototype = trung bình vector các phim train thuộc thể loại, mỗi phim góp một lần"""
    vectors, labels, counts = toy()
    prototypes, support = P.label_prototypes(vectors, labels, counts)
    assert support.tolist() == [2, 2]
    assert np.allclose(prototypes[0], [0.5, 1.0]) and np.allclose(prototypes[1], [1.5, 2.5])


def test_label_weights_sum_to_one():
    """Tổng trọng số trên các nhãn hợp lệ bằng 1; phim không nhãn có trọng số 0"""
    _, labels, counts = toy()
    weights = P.uniform_label_weights(labels, np.array([2, 2]))
    assert np.allclose(weights[:3].sum(axis=1), 1.0) and weights[3].sum() == 0


def test_scaling_and_fallback():
    """Vector taxonomy có độ lớn = trung vị độ lớn phim train; phim không nhãn quay về CF"""
    vectors, labels, counts = toy()
    taxonomy, has_valid, _ = P.taxonomy_vectors(vectors, labels, counts)
    median = np.median(np.linalg.norm(vectors, axis=1))
    assert np.allclose(np.linalg.norm(taxonomy[has_valid], axis=1), median)
    assert has_valid.tolist() == [True, True, True, False]
    mixed = P.mix(vectors, taxonomy, 0.0, has_valid)
    assert np.allclose(mixed[3], vectors[3]) and not np.isnan(mixed).any()


def test_mix_endpoints():
    """g = 1 khôi phục điểm CF; g = 0 chỉ còn điểm prototype"""
    vectors, labels, counts = toy()
    taxonomy, has_valid, _ = P.taxonomy_vectors(vectors, labels, counts)
    user = np.array([0.3, -0.7])
    assert np.allclose(P.mix(vectors, taxonomy, 1.0, has_valid) @ user, vectors @ user)
    only = P.mix(vectors, taxonomy, 0.0, has_valid) @ user
    assert np.allclose(only[has_valid], taxonomy[has_valid] @ user)
    assert np.allclose(P.mix_scores(vectors @ user, taxonomy @ user, 1.0), vectors @ user)
