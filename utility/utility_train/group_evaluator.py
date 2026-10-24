"""Chỉ số Recall@K, NDCG@K, Precision@K theo người dùng và theo nhóm phim (bộ đề, trang 9).

Quy tắc:
- Xếp hạng TOÀN BỘ danh mục; nhóm chỉ thay tập đích đúng T bằng các đích thuộc nhóm, không đổi bảng xếp hạng.
- Người dùng không có đích thuộc nhóm không tham gia trung bình của nhóm.
- NDCG nhóm dùng IDCG theo số đích của chính nhóm.
- Không cộng trung bình các nhóm để suy ra chỉ số toàn bộ.
- Nhóm có dưới 30 người dùng hoặc dưới 100 đích được gắn nhãn "thăm dò" (ngưỡng thực hành của bộ đề).

Ví dụ tự kiểm tra (trang 9): đích {A, B}, top-3 = [X, A, B] → Recall@3 = 1; NDCG@3 ≈ 0,6934;
nếu chỉ A thuộc nhóm ít tương tác thì Recall nhóm = 1 và NDCG nhóm ≈ 0,6309.
"""

import json
import math
from pathlib import Path

EXPLORATORY_MIN_USERS = 30
EXPLORATORY_MIN_TARGETS = 100


def user_recall(ranking, positives, k):
    positives = set(positives)
    return sum(item in positives for item in ranking[:k]) / float(len(positives))


def user_precision(ranking, positives, k):
    positives = set(positives)
    return sum(item in positives for item in ranking[:k]) / float(k)


def user_ndcg(ranking, positives, k):
    positives = set(positives)
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(ranking[:k]) if item in positives)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(positives), k)))
    return dcg / idcg if idcg else 0.0


def _normalize(targets):
    return {int(user): [int(i) for i in items] for user, items in targets.items() if items}


def support(targets):
    targets = _normalize(targets)
    return {"users": len(targets), "targets": sum(len(v) for v in targets.values()),
            "items": len({i for v in targets.values() for i in v})}


def is_exploratory(group_support):
    return group_support["users"] < EXPLORATORY_MIN_USERS or group_support["targets"] < EXPLORATORY_MIN_TARGETS


def evaluate_slice(rankings, targets, ks=(10, 20), keep_per_user=False):
    """Trung bình chỉ số trên các user có ít nhất một đích trong `targets`."""
    targets = _normalize(targets)
    missing = [user for user in targets if user not in rankings]
    if missing:
        raise ValueError("Thiếu bảng xếp hạng của user: {}".format(missing[:10]))
    users = sorted(targets)
    result = {"support": support(targets)}
    result["exploratory"] = is_exploratory(result["support"])
    per_user = {}
    for metric, function in (("recall", user_recall), ("ndcg", user_ndcg), ("precision", user_precision)):
        result[metric] = {}
        for k in ks:
            values = [function(rankings[user], targets[user], k) for user in users]
            result[metric][str(k)] = sum(values) / len(values) if values else 0.0
            if keep_per_user:
                for user, value in zip(users, values):
                    per_user.setdefault(user, {})["{}@{}".format(metric, k)] = value
    if keep_per_user:
        result["per_user"] = per_user
    return result


def evaluate_rankings(rankings, targets_by_group, ks=(10, 20), keep_per_user=False):
    """targets_by_group: {"overall": {...}, "<nhóm>": {...}} → {nhóm: kết quả}."""
    return {group: evaluate_slice(rankings, targets, ks, keep_per_user) for group, targets in targets_by_group.items()}


def load_protocol(protocol_directory, split):
    """Đọc định nghĩa nhóm và tập đích theo nhóm do tools/protocol/build_protocol.py sinh ra."""
    directory = Path(protocol_directory)
    path = directory / "{}_targets_by_group.json".format(split)
    if not path.is_file():
        raise FileNotFoundError("Thiếu {}. Chạy trước: python -m tools.data.prepare".format(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    definitions = json.loads((directory / "group_definitions.json").read_text(encoding="utf-8"))
    targets = {group: {int(u): items for u, items in payload[group].items()} for group in payload}
    return definitions, targets


def result_rows(result, ks=(10, 20), definitions=None):
    """Chuyển kết quả thành các dòng CSV (mỗi nhóm một dòng)."""
    rows = []
    for group, values in result.items():
        row = {"group": group, "label": (definitions or {}).get(group, {}).get("label", group),
               "users": values["support"]["users"], "targets": values["support"]["targets"],
               "items": values["support"]["items"], "exploratory": int(values["exploratory"])}
        for metric in ("recall", "ndcg", "precision"):
            for k in ks:
                row["{}@{}".format(metric, k)] = values[metric][str(k)]
        rows.append(row)
    return rows
