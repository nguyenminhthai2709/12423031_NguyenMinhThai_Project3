"""Kiểm thử lõi: mô hình nền do GV cấp (chỉ kiểm tra những mô hình có trong core_manifest.json)."""

import importlib
import json
from pathlib import Path

import pytest
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]


def core_models():
    manifest = json.loads((ROOT / "core_manifest.json").read_text(encoding="utf-8"))
    return [Path(name).stem for name in manifest["files"] if name.startswith("models/") and not name.endswith("__init__.py")]


class ToyData:
    def __init__(self):
        self.num_users, self.num_items = 3, 4
        self.num_nodes = 7
        users, items = [0, 0, 1, 2, 2], [0, 1, 1, 2, 3]
        self.user_item_net = sp.csr_matrix(([1.0] * 5, (users, items)), shape=(3, 4))


CONFIG = {"reg_lambda": "0.0001", "embedding_size": "8", "GCN_layer": "1"}


@pytest.mark.parametrize("name", ["BPRMF", "LightGCN"])
def test_model_scores_are_dot_products(name):
    """Điểm chấm = tích vô hướng vector user và vector phim; hàm mất mát trả về [BPR, phạt L2] hữu hạn"""
    if name not in core_models():
        pytest.skip("KHÔNG ÁP DỤNG: mô hình {} không thuộc lõi của đề tài này".format(name))
    module = importlib.import_module("models." + name)
    model = getattr(module, name)(CONFIG, ToyData(), torch.device("cpu"))
    users, items = model.aggregate()
    scores = model.get_rating_for_test(torch.tensor([0, 2]))
    assert torch.allclose(scores, users[[0, 2]] @ items.T)
    losses = model(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([3, 0]))
    assert len(losses) == 2 and all(torch.isfinite(value) for value in losses)


def test_lightgcn_graph_uses_train_only():
    """Đồ thị LightGCN chỉ có cạnh của train (số cạnh = 2 x số tương tác train)"""
    if "LightGCN" not in core_models():
        pytest.skip("KHÔNG ÁP DỤNG: LightGCN không thuộc lõi của đề tài này")
    from utility.utility_data.data_graph import bipartite_adjacency
    assert bipartite_adjacency(ToyData()).nnz == 10
