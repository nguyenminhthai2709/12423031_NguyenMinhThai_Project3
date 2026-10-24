"""Thí nghiệm K01 (phương án 1 SV): CF gốc, prototype lá và hỗn hợp lá–cha trên cây quy ước (BPR-MF nền, 2 seed).

Bước 1:  python main.py --model BPRMF --seeds 42 0 1 --tag base
Bước 2:  python -m tools.experiments.run_K01
Quy trình: g = 0,5 cố định cho mọi biến thể taxonomy (không chọn g); cùng checkpoint, cùng tập phim train.
Đầu ra: results/K01_validation.csv, results/K01_test.csv, results/K01_tree_coverage.csv, results/K01_node_support.csv.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import PROJECT_ROOT  # noqa: E402
from models import K01_two_level as K  # noqa: E402
from tools.experiments import common  # noqa: E402
from utility.utility_function import prototype as P  # noqa: E402
from utility.utility_train import posthoc  # noqa: E402


def main():
    config = common.topic_config("K01")
    parent_of, parent_ids = K.load_tree(PROJECT_ROOT / config.get("tree_file", "metadata/convention_tree.json"))
    methods = common.json_value(config, "methods", ["CF", "leaf", "mix"])
    g = float(config.get("mix_g", 0.5))
    validation_rows, test_rows, coverage_rows, support_rows = [], [], [], []
    for seed in common.json_value(config, "seeds", [42, 0, 1]):
        bundle = common.load_base("BPRMF", seed, tag=config.get("base_tag", "base"))
        dataset = bundle.dataset
        users, items = posthoc.frozen_embeddings(bundle)
        labels, names = common.item_labels(dataset)
        counts = dataset.item_train_counts()
        membership = K.parent_membership(labels, names, parent_of, parent_ids)
        if not support_rows:
            _, leaf_support = P.label_prototypes(items, labels, counts)
            _, parent_support = P.label_prototypes(items, membership, counts)
            for name, value in zip(names, leaf_support):
                support_rows.append({"level": "leaf", "node": name, "parent": parent_of[name],
                                     "train_items": int(value)})
            for name, value in zip(parent_ids, parent_support):
                support_rows.append({"level": "parent", "node": name, "parent": "", "train_items": int(value)})
            coverage_rows.append({"items": int(dataset.num_items), "leaf_nodes": len(names),
                                  "parent_nodes": len(parent_ids), "root_used": 0,
                                  "items_without_leaf": int((labels.sum(axis=1) == 0).sum()),
                                  "items_multi_parent": int((membership.sum(axis=1) > 1).sum()),
                                  "items_two_leaves_same_parent": int((labels.sum(axis=1) > membership.sum(axis=1)).sum())})
        for method in methods:
            if method == "CF":
                vectors = items
            else:
                taxonomy, valid = K.two_level_vectors(items, labels, membership, counts, method)
                vectors = P.mix(items, taxonomy, g, valid)
            for split, rows in (("validation", validation_rows), ("test", test_rows)):
                result, definitions = common.evaluate(dataset, config, common.dot_scores(users, vectors), split)
                rows += common.rows(result, definitions, config, seed=seed, method=method,
                                    g=(1.0 if method == "CF" else g), checkpoint=str(bundle.run_dir))
    common.write_csv("K01_validation.csv", validation_rows)
    common.write_csv("K01_test.csv", test_rows)
    common.write_csv("K01_tree_coverage.csv", coverage_rows)
    common.write_csv("K01_node_support.csv", support_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
