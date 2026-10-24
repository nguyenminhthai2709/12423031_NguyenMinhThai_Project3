"""Giao thức G2 (bộ đề, trang 10): prototype thể loại và bộ trộn trên vector đã đóng băng.

1. Prototype của thể loại c = trung bình vector các phim có số đếm train > 0 thuộc c; mỗi phim góp đúng một lần.
   Một phim đa nhãn góp vào nhiều thể loại. (Prototype chứa vector của chính phim train: đây là quy ước tự
   đóng góp, không phải rò rỉ test, nhưng phải ghi rõ trong báo cáo.)
2. Vector taxonomy của phim = trung bình (đều) các prototype nhãn hợp lệ của phim. Các đề tài D07, D08, D10
   chỉ thay đúng thao tác được chỉ định; phần còn lại giữ nguyên.
3. Chuẩn hóa vector taxonomy về độ dài 1 rồi nhân với TRUNG VỊ độ lớn vector phim train của checkpoint nền
   (epsilon = 1e-8). Hệ số này dùng chung cho mọi nhánh so sánh.
4. Phim không còn nhãn hợp lệ hoặc vector tổng gần 0: quay về CF (trọng số CF = 1).
5. Bộ trộn: e' = g * e_CF + (1 - g) * v_taxonomy; điểm = <e_u, e'_i>. Với cổng phụ thuộc người dùng (K02) dùng
   trộn điểm tương đương: s' = g * s_CF + (1 - g) * s_taxonomy.
Kiểm tra tối thiểu: g = 1 khôi phục điểm CF; g = 0 chỉ còn điểm prototype; tổng trọng số nhãn bằng 1; không NaN.
"""

import numpy as np

EPSILON = 1e-8


def label_prototypes(item_vectors, item_labels, train_counts):
    """Trả về (prototypes [L x d], support [L]). support[c] = số phim train thuộc c; thể loại không có phim: hàng 0."""
    item_vectors = np.asarray(item_vectors, dtype=np.float64)
    labels = np.asarray(item_labels, dtype=bool) & (np.asarray(train_counts) > 0)[:, None]
    support = labels.sum(axis=0).astype(np.int64)
    sums = labels.T.astype(np.float64) @ item_vectors
    prototypes = np.zeros_like(sums)
    active = support > 0
    prototypes[active] = sums[active] / support[active, None]
    return prototypes, support


def uniform_label_weights(item_labels, support):
    """Trọng số gộp đều trên các nhãn hợp lệ (support > 0) của từng phim; hàng không có nhãn hợp lệ toàn 0."""
    valid = np.asarray(item_labels, dtype=bool) & (np.asarray(support) > 0)[None, :]
    counts = valid.sum(axis=1, keepdims=True)
    weights = np.divide(valid.astype(np.float64), counts, out=np.zeros(valid.shape, dtype=np.float64), where=counts > 0)
    return weights


def pool_items(prototypes, weights):
    """Vector taxonomy chưa cân thang của từng phim = Σ_c w[i, c] * p_c."""
    return np.asarray(weights, dtype=np.float64) @ np.asarray(prototypes, dtype=np.float64)


def median_train_norm(item_vectors, train_counts):
    norms = np.linalg.norm(np.asarray(item_vectors, dtype=np.float64), axis=1)
    return float(np.median(norms[np.asarray(train_counts) > 0]))


def scale_like_items(taxonomy_vectors, item_vectors, train_counts):
    """Chuẩn hóa về độ dài 1 rồi nhân trung vị độ lớn vector phim train. Trả về (vector, has_valid)."""
    vectors = np.asarray(taxonomy_vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    has_valid = norms > EPSILON
    scaled = np.zeros_like(vectors)
    scaled[has_valid] = vectors[has_valid] / (norms[has_valid, None] + EPSILON)
    return scaled * median_train_norm(item_vectors, train_counts), has_valid


def taxonomy_vectors(item_vectors, item_labels, train_counts):
    """Toàn bộ G2 mặc định: prototype → gộp đều → cân thang. Trả về (vector taxonomy, has_valid, support)."""
    prototypes, support = label_prototypes(item_vectors, item_labels, train_counts)
    pooled = pool_items(prototypes, uniform_label_weights(item_labels, support))
    scaled, has_valid = scale_like_items(pooled, item_vectors, train_counts)
    return scaled, has_valid, support


def mix(item_vectors, taxonomy, g, has_valid):
    """e' = g * e + (1 - g) * v; phim không có nhãn hợp lệ giữ nguyên e (g = 1). g: số hoặc mảng [số phim]."""
    item_vectors = np.asarray(item_vectors, dtype=np.float64)
    g = np.broadcast_to(np.asarray(g, dtype=np.float64), (item_vectors.shape[0],)).copy()
    if np.any((g < 0) | (g > 1)):
        raise ValueError("g phải nằm trong [0, 1]")
    g[~np.asarray(has_valid, dtype=bool)] = 1.0
    return g[:, None] * item_vectors + (1.0 - g[:, None]) * np.asarray(taxonomy, dtype=np.float64)


def mix_scores(cf_scores, taxonomy_scores, g):
    """s' = g * s_CF + (1 - g) * s_taxonomy (g có thể là ma trận cùng kích thước, ví dụ cổng theo user–phim)."""
    g = np.asarray(g, dtype=np.float64)
    return g * np.asarray(cf_scores, dtype=np.float64) + (1.0 - g) * np.asarray(taxonomy_scores, dtype=np.float64)


def global_prototype(item_vectors, train_counts):
    """Trung bình toàn cục của vector phim train (đối chứng "làm trơn thuần túy")."""
    item_vectors = np.asarray(item_vectors, dtype=np.float64)
    return item_vectors[np.asarray(train_counts) > 0].mean(axis=0)
