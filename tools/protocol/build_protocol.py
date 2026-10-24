"""Sinh giao thức đánh giá theo nhóm phim từ số tương tác TRAIN (bộ đề, trang 9).

Nhóm (định nghĩa chỉ dùng train, không dùng validation/test):
    very_few_1_5   : 1–5 tương tác train  (nhóm rất ít; tập con của nhóm 1–10; trùng ngưỡng Near-Cold của NCKH/TaxPro-CL)
    long_tail_1_10 : 1–10 tương tác train (nhóm trọng tâm)
    warm_gt10      : > 10 tương tác train (đối chiếu đánh đổi)
    cold_0         : 0 tương tác train    (chỉ xuất hiện khi split có phim cold, ví dụ giao thức C1 của T05)
    warm_1plus     : >= 1 tương tác train (đi kèm cold_0; phần warm của bảng hỗn hợp C1)

Đầu ra trong preprocessed/evaluation_protocol/<dataset>/: group_definitions.json, item_degrees_train.json,
validation_targets_by_group.json, test_targets_by_group.json, group_statistics.json, manifest.json.

    python -m tools.protocol.build_protocol                 # cho split G1 (ml-100k hoặc P3_DATASET)
    python -m tools.protocol.build_protocol --dataset ml-100k-c1
    python -m tools.protocol.build_protocol --dataset amazon-book --group_set nc_lt_wm   # đề tài NCKH
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import (DATASET_NAME, dataset_split_file, listed_item_count, protocol_dir,  # noqa: E402
                                     verified_dataset_dir)

GROUPS = {
    # Lõi 1.2: nhóm rất ít đổi 1–2 → 1–5. Sau 5-core (giao thức B) nhóm 1–2 của ML-100K chỉ còn 11 phim / 27 đích
    # test (luôn "thăm dò"); 1–5 có 131 phim / 205 đích test. Bảng căn cứ: docs/HUONG_DAN_CHUNG.md, mục nhóm phim.
    "very_few_1_5": {"label": "Rất ít (1–5)", "minimum": 1, "maximum": 5},
    "long_tail_1_10": {"label": "Ít / long-tail (1–10)", "minimum": 1, "maximum": 10},
    "warm_gt10": {"label": "Nhiều hơn (> 10)", "minimum": 11, "maximum": None},
}
GROUP_SETS = {
    "g1": GROUPS,
    # Đề tài NCKH (Amazon-Book): Near-Cold 1–5, Long-Tail 1–10, Warm > 10, đều tính theo số tương tác train
    "nc_lt_wm": {
        "near_cold_1_5": {"label": "Near-Cold, NC (1–5)", "minimum": 1, "maximum": 5},
        "long_tail_1_10": {"label": "Long-Tail, LT (1–10)", "minimum": 1, "maximum": 10},
        "warm_gt10": {"label": "Warm, Wm (> 10)", "minimum": 11, "maximum": None},
    },
}
COLD_GROUP = {"cold_0": {"label": "Cold (0)", "minimum": 0, "maximum": 0},
              "warm_1plus": {"label": "Warm (>= 1)", "minimum": 1, "maximum": None}}


def read_split(path):
    part = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            values = [int(v) for v in line.split()]
            if len(values) > 1:
                part[values[0]] = values[1:]
    return part


def in_group(degree, definition):
    if degree < definition["minimum"]:
        return False
    return definition["maximum"] is None or degree <= definition["maximum"]


def build(dataset_name, group_set="g1"):
    directory = verified_dataset_dir(dataset_name)
    train = read_split(dataset_split_file(directory, "train"))
    validation = read_split(dataset_split_file(directory, "validation"))
    test = read_split(dataset_split_file(directory, "test"))
    num_items = max(1 + max(max(items) for part in (train, validation, test) for items in part.values()),
                    listed_item_count(directory))
    degrees = [0] * num_items
    for items in train.values():
        for item in items:
            degrees[item] += 1
    definitions = dict(GROUP_SETS[group_set])
    if any(degree == 0 for degree in degrees):
        definitions.update(COLD_GROUP)
    output = protocol_dir(dataset_name)
    output.mkdir(parents=True, exist_ok=True)
    statistics = {"num_items": num_items, "groups": {}}
    for split, part in (("validation", validation), ("test", test)):
        targets = {"overall": {str(u): items for u, items in sorted(part.items())}}
        for group, definition in definitions.items():
            sliced = {}
            for user, items in sorted(part.items()):
                kept = [item for item in items if in_group(degrees[item], definition)]
                if kept:
                    sliced[str(user)] = kept
            targets[group] = sliced
        (output / "{}_targets_by_group.json".format(split)).write_bytes(json.dumps(targets).encode("utf-8"))
        for group, sliced in targets.items():
            entry = statistics["groups"].setdefault(group, {})
            entry[split] = {"users": len(sliced), "targets": sum(len(v) for v in sliced.values())}
    for group, definition in definitions.items():
        statistics["groups"][group]["catalog_items"] = sum(in_group(d, definition) for d in degrees)
    (output / "group_definitions.json").write_bytes(json.dumps(definitions, ensure_ascii=False, indent=2).encode("utf-8"))
    (output / "item_degrees_train.json").write_bytes(json.dumps({"degrees": degrees}).encode("utf-8"))
    (output / "group_statistics.json").write_bytes(json.dumps(statistics, ensure_ascii=False, indent=2).encode("utf-8"))
    hashes = {split: hashlib.sha256(dataset_split_file(directory, split).read_bytes()).hexdigest()
              for split in ("train", "validation", "test")}
    manifest = {"dataset": dataset_name, "group_set": group_set, "group_source": "train degree only",
                "split_sha256": hashes}
    (output / "manifest.json").write_bytes(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    return statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--group_set", choices=sorted(GROUP_SETS), default="g1")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.dataset, args.group_set), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
