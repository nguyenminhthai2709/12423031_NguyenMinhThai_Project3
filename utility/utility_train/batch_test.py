"""Đánh giá xếp hạng: chọn checkpoint trên validation, mở test một lần.

- Ứng viên = toàn bộ danh mục G1. Validation loại dương train; test loại thêm dương validation (không loại đích test).
- Phim bị loại nhận điểm -inf (không bao giờ lọt vào top-K).
- Phá hòa điểm bằng một hoán vị mã phim cố định (seed riêng `tiebreak_seed`), dùng chung cho mọi phương pháp.
- Chọn checkpoint theo `selection_metric`@`selection_K` trên validation TOÀN BỘ (mặc định NDCG@10);
  `selection_group` cho phép chọn trên một nhóm khác (chỉ dùng cho split C1: warm_1plus).
- `candidate_policy = train_catalog` chỉ xếp hạng phim có tương tác train (mô hình nền trên split C1);
  mặc định `all` xếp hạng mọi phim của split.

Mọi bộ chấm điểm (mô hình học được, Pop, GenreProfile, prototype…) đều đi qua `evaluate_scores` để dùng
chung ứng viên, mặt nạ, phá hòa và cách tính chỉ số.
"""

import csv
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

from config_path.config_path import (PROJECT_ROOT, checkpoint_file, protocol_dir, result_file, temporary_file)
from utility.utility_train.group_evaluator import evaluate_rankings, load_protocol, result_rows


def parse_ks(config):
    value = config.get("top_K", "[10, 20]")
    return tuple(int(v) for v in json.loads(value)) if isinstance(value, str) else tuple(int(v) for v in value)


def tie_break_permutation(num_items, seed):
    return np.random.RandomState(int(seed)).permutation(int(num_items))


def rank_top_k(scores, exclude_rows, k, permutation):
    """scores: tensor [B x I]; exclude_rows: list mảng mã phim cần loại cho từng user; trả về np [B x k].

    Xếp giảm dần theo điểm; khi bằng điểm, phim đứng trước trong `permutation` được xếp trước.
    """
    scores = torch.as_tensor(scores, dtype=torch.float32).clone()
    lengths = [len(items) for items in exclude_rows]
    if sum(lengths):
        # một phép gán cho cả lô (nhanh hơn nhiều so với gán từng user, nhất là trên GPU)
        rows = np.repeat(np.arange(len(exclude_rows), dtype=np.int64), lengths)
        cols = np.concatenate([np.asarray(items, dtype=np.int64) for items in exclude_rows if len(items)])
        scores[torch.as_tensor(rows, device=scores.device), torch.as_tensor(cols, device=scores.device)] = float("-inf")
    perm = torch.as_tensor(permutation, dtype=torch.long, device=scores.device)
    reordered = scores[:, perm]
    _, order = torch.sort(reordered, dim=1, descending=True, stable=True)
    return perm[order[:, :k]].cpu().numpy()


def protocol_directory(config):
    explicit = config.get("evaluation_protocol_path")
    if explicit:
        path = Path(str(explicit))
        return path if path.is_absolute() else PROJECT_ROOT / path
    return protocol_dir(config["dataset"])


def compute_rankings(dataset, score_fn, split, k, batch_size, tiebreak_seed, candidate_policy="all"):
    """score_fn(np.ndarray mã user) -> tensor/np điểm [B x số phim]. Trả về {user: [top-k mã phim]}."""
    if candidate_policy not in ("all", "train_catalog"):
        raise ValueError("candidate_policy phải là all hoặc train_catalog")
    untrained = np.flatnonzero(dataset.item_train_counts() == 0) if candidate_policy == "train_catalog" else []
    targets = dataset.validation_dict if split == "validation" else dataset.test_dict
    users = sorted(targets)
    permutation = tie_break_permutation(dataset.num_items, tiebreak_seed)
    rankings = {}
    for start in range(0, len(users), batch_size):
        batch = np.asarray(users[start:start + batch_size], dtype=np.int64)
        with torch.no_grad():
            scores = score_fn(batch)
        exclude = dataset.get_user_pos_items_for_evaluation(batch, split)
        if len(untrained):
            exclude = [np.union1d(items, untrained) for items in exclude]
        top = rank_top_k(scores, exclude, k, permutation)
        for user, ranked in zip(batch.tolist(), top.tolist()):
            rankings[int(user)] = ranked
    return rankings


def evaluate_scores(dataset, score_fn, config, split, keep_per_user=False):
    """Đánh giá một bộ chấm điểm bất kỳ trên validation hoặc test, toàn bộ và theo nhóm."""
    ks = parse_ks(config)
    rankings = compute_rankings(dataset, score_fn, split, max(ks), int(config.get("test_batch_size", 1024)),
                                int(config.get("tiebreak_seed", 12345)), config.get("candidate_policy", "all"))
    definitions, targets = load_protocol(protocol_directory(config), split)
    result = evaluate_rankings(rankings, targets, ks, keep_per_user)
    return result, definitions


def model_score_fn(model, device):
    model.eval()

    def score(users):
        return model.get_rating_for_test(torch.as_tensor(users, dtype=torch.long, device=device))
    return score


def _selection(config, result):
    metric = str(config.get("selection_metric", "ndcg")).lower()
    k = str(int(config.get("selection_K", 10)))
    group = str(config.get("selection_group", "overall"))
    if metric not in ("ndcg", "recall"):
        raise ValueError("selection_metric phải là ndcg hoặc recall")
    if group not in result:
        raise ValueError("selection_group {} không có trong giao thức".format(group))
    name = "{}@{}".format(metric, k) if group == "overall" else "{}@{}[{}]".format(metric, k, group)
    return name, float(result[group][metric][k])


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(path)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def save_checkpoint(path, model, optimizer, epoch, metrics, config, selection_name, selection_value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(path)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "validation_metrics": metrics, "configuration": dict(config),
                "selection_metric": selection_name, "selection_value": selection_value,
                "training_seed": getattr(model, "training_seed", None),
                "rng_state": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}},
               temporary)
    os.replace(str(temporary), str(path))


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _compact(result):
    return {group: {key: value for key, value in values.items() if key != "per_user"} for group, values in result.items()}


def general_test(dataset, model, device, config, epoch, best, optimizer=None, logger=None):
    """Đánh giá validation, cập nhật dừng sớm, lưu checkpoint tốt nhất và checkpoint cuối."""
    logger = logger or logging.getLogger(__name__)
    result, _ = evaluate_scores(dataset, model_score_fn(model, device), config, "validation")
    name, value = _selection(config, result)
    dataset.validation_history.append({"epoch": epoch + 1, "selection": name, "value": value, "metrics": _compact(result)})
    output_dir = Path(dataset.training_output_dir)
    _write_json_atomic(result_file(output_dir, "validation"), dataset.validation_history)
    if best.get("epoch", 0) == 0 or value > best.get("value", float("-inf")):
        best.update(count=0, epoch=epoch + 1, value=value, name=name)
        save_checkpoint(checkpoint_file(output_dir, "best_validation"), model, optimizer, epoch + 1, _compact(result),
                        config, name, value)
        logger.info("Validation tốt nhất mới: epoch=%d %s=%.6f", epoch + 1, name, value)
    else:
        best["count"] = best.get("count", 0) + 1
        if best["count"] >= int(config.get("early_stopping", 4)):
            best["stop"] = True
            logger.info("Dừng sớm: %d lần kiểm tra không cải thiện (tốt nhất epoch %d)", best["count"], best["epoch"])
    save_checkpoint(checkpoint_file(output_dir, "last"), model, optimizer, epoch + 1, _compact(result), config, name, value)
    logger.info("Validation epoch %d: %s=%.6f", epoch + 1, name, value)
    return result, best


def write_group_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_per_user_csv(path, result, ks):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        columns = ["{}@{}".format(m, k) for m in ("recall", "ndcg") for k in ks]
        writer.writerow(["group", "user"] + columns)
        for group, values in result.items():
            for user, metrics in sorted(values.get("per_user", {}).items()):
                writer.writerow([group, user] + [metrics[c] for c in columns])


def final_test(dataset, model, device, config, logger=None):
    """Nạp checkpoint tốt nhất trên validation rồi đánh giá test đúng một lần."""
    logger = logger or logging.getLogger(__name__)
    output_dir = Path(dataset.training_output_dir)
    checkpoint = load_checkpoint(checkpoint_file(output_dir, "best_validation"), device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    ks = parse_ks(config)
    result, definitions = evaluate_scores(dataset, model_score_fn(model, device), config, "test", keep_per_user=True)
    _write_json_atomic(result_file(output_dir, "final_test"),
                       {"best_validation_epoch": checkpoint.get("epoch"), "selection_metric": checkpoint.get("selection_metric"),
                        "selection_value": checkpoint.get("selection_value"), "test_metrics": _compact(result)})
    write_group_csv(result_file(output_dir, "final_test_groups"), result_rows(result, ks, definitions))
    write_per_user_csv(output_dir / "final_test_per_user.csv", result, ks)
    logger.info("Test (epoch %s): %s", checkpoint.get("epoch"),
                {g: round(v["ndcg"][str(ks[0])], 4) for g, v in result.items()})
    return result
