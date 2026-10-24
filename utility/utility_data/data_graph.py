"""Ma trận kề chuẩn hóa theo bậc D^-1/2 A D^-1/2 của đồ thị hai phía user–phim, dựng CHỈ từ train.

Khác bản gốc ID-GRec: không lưu cache pre_A.npz vào thư mục dữ liệu. Với ML100K việc dựng lại mất dưới
1 giây, và bỏ cache tránh lỗi dùng nhầm đồ thị của một split cũ sau khi dựng lại dữ liệu.
"""

import logging

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def bipartite_adjacency(data):
    """[[0, R], [R^T, 0]] với R là ma trận tương tác train (users x items)."""
    interaction = data.user_item_net.tocsr().astype(np.float32, copy=False)
    adjacency = sp.bmat([[None, interaction], [interaction.T, None]], format="csr", dtype=np.float32)
    if adjacency.shape != (data.num_nodes, data.num_nodes) or adjacency.nnz != 2 * interaction.nnz:
        raise ValueError("Ma trận kề không khớp số nút / số cạnh train")
    return adjacency


def normalized_adjacency(data):
    adjacency = bipartite_adjacency(data)
    row_sum = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    with np.errstate(divide="ignore"):
        d_inv = np.power(row_sum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.0
    degree = sp.diags(d_inv)
    norm = degree.dot(adjacency).dot(degree).tocsr()
    logger.info("Đồ thị chuẩn hóa: %d nút, %d cạnh (chỉ từ train)", norm.shape[0], adjacency.nnz // 2)
    return norm
