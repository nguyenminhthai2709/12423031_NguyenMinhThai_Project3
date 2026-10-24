"""Dùng lại một run đã huấn luyện: nạp checkpoint tốt nhất, lấy vector đóng băng, chấm điểm lại.

Các đề tài làm việc trên "vector đóng băng" (prototype, bộ trộn, cổng…) dùng module này để mọi biến thể
được chấm trên CÙNG checkpoint, cùng ứng viên, cùng phá hòa và cùng cách tính chỉ số.

Ví dụ:
    from utility.utility_train import posthoc
    run = posthoc.latest_run("BPRMF", seed=42)
    bundle = posthoc.load_run(run)                      # model, dataset, config
    U, V = posthoc.frozen_embeddings(bundle)            # numpy [users x d], [items x d]
    result, defs = posthoc.evaluate_item_vectors(bundle, U, V_moi, "validation")
"""

import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from config_path.config_path import DATASET_NAME, LOG_DIR, checkpoint_file, resolve_dataset, result_file, verified_dataset_dir
from utility.utility_data.data_loader import Data
from utility.utility_train import batch_test


def completed_runs(model_name, dataset_name=DATASET_NAME, tag=None):
    """Danh sách thư mục run đã hoàn tất (run_manifest.json có status = completed)."""
    root = LOG_DIR / model_name / resolve_dataset(dataset_name)
    runs = []
    for manifest in sorted(root.glob("*/seed*/run_manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") == "completed" and (tag is None or payload.get("tag") == tag):
            runs.append(manifest.parent)
    return runs


def latest_run(model_name, seed, dataset_name=DATASET_NAME, tag=None):
    runs = [run for run in completed_runs(model_name, dataset_name, tag) if run.name == "seed{}".format(seed)]
    if not runs:
        raise FileNotFoundError("Chưa có run hoàn tất của {} seed {} (tag={}). Chạy main.py trước.".format(
            model_name, seed, tag))
    return max(runs, key=lambda path: path.stat().st_mtime)


def load_run(run_directory, device="cpu"):
    run_directory = Path(run_directory)
    manifest = json.loads(result_file(run_directory, "manifest").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("{} chưa hoàn tất".format(run_directory))
    config = manifest["configuration"]
    device = torch.device(device)
    logger = logging.getLogger(__name__)
    dataset = Data(str(verified_dataset_dir(config["dataset"])), config, logger=logger)
    dataset.training_output_dir = run_directory
    trainer_class = getattr(importlib.import_module("models." + manifest["model"]), "Trainer")
    model = trainer_class(SimpleNamespace(seed=manifest.get("seed")), config, dataset, device, logger).model
    checkpoint = batch_test.load_checkpoint(checkpoint_file(run_directory, "best_validation"), device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return SimpleNamespace(model=model, dataset=dataset, config=config, run_dir=run_directory,
                           seed=manifest.get("seed"), model_name=manifest["model"], device=device)


def frozen_embeddings(bundle):
    """Vector dùng để chấm điểm của checkpoint (MF: vector đầu vào; LightGCN: sau trung bình các lớp)."""
    with torch.no_grad():
        users, items = bundle.model.aggregate()
    return users.detach().cpu().numpy().astype(np.float64), items.detach().cpu().numpy().astype(np.float64)


def evaluate_scores(bundle, score_fn, split, keep_per_user=False):
    """score_fn(np.ndarray mã user) -> np/tensor [B x số phim]."""
    return batch_test.evaluate_scores(bundle.dataset, score_fn, bundle.config, split, keep_per_user)


def evaluate_item_vectors(bundle, user_vectors, item_vectors, split, keep_per_user=False):
    """Chấm điểm bằng tích vô hướng <u, v_i> với vector phim mới (ví dụ vector đã trộn prototype)."""
    users = torch.as_tensor(np.asarray(user_vectors), dtype=torch.float64)
    items = torch.as_tensor(np.asarray(item_vectors), dtype=torch.float64)
    if items.shape[0] != bundle.dataset.num_items:
        raise ValueError("item_vectors phải có đúng {} hàng".format(bundle.dataset.num_items))

    def score(batch):
        return (users[torch.as_tensor(batch, dtype=torch.long)] @ items.T).float()
    return evaluate_scores(bundle, score, split, keep_per_user)
