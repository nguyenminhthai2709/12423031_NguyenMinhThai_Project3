"""Dựng split G1 từ MovieLens 100K theo giao thức B của TaxPro-CL (dữ liệu chưa có split sẵn).

Quy ước của dự án: dữ liệu đã có split công khai (Amazon-Book của LightGCN) dùng giao thức A – giữ test công khai,
lọc 5-core train, tách validation; dữ liệu chưa có split (MovieLens 100K) dùng giao thức B dưới đây.

1. rating >= 4 → tương tác dương (rating thấp hơn không vào tập dương, không mặc định là nhãn âm chắc chắn).
2. Khử trùng cặp user–phim TRƯỚC khi chia; nếu lặp, giữ bản ghi có thời gian muộn nhất.
3. Lọc 5-core LẶP trên cả user và phim (mỗi user và mỗi phim có >= 5 tương tác dương) cho tới khi ổn định.
4. Với mỗi user (duyệt theo mã gốc tăng dần, MỘT bộ sinh Python random.Random(42) cho toàn bộ quá trình): sắp các phim
   tăng dần, trộn, rồi lấy round(0,7n) cho train, round(0,1n) cho validation, phần còn lại cho test (round của Python
   là làm tròn ngân hàng). Không lọc k-core lần hai sau khi chia. Giống `per_user_split` của TaxPro-CL.
5. Danh mục chỉ gồm phim có >= 1 tương tác train. Đích validation/test thuộc phim ngoài danh mục (nếu có) được tách ra
   và đếm rõ.
6. Ánh xạ mã liên tiếp: user và phim (theo mã gốc tăng dần) → 0, 1, 2, ...

Đầu ra: dataset_verify/<dataset>/{train,validation,test}.txt, user_list.txt, item_list.txt, split_manifest.json;
metadata/<dataset>/item_genres.csv và genre_names.json (18 thể loại, bỏ nhãn `unknown`). <dataset> = ml-100k (mặc định)
hoặc ml-1m (P3_DATASET=ml-1m; u.data/u.item/u.genre đã được download_ml100k chuẩn hóa từ ratings.dat/movies.dat).

    python -m tools.data.build_g1
"""

import argparse
import collections
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import (DATASET_NAME, METADATA_DIR, raw_dataset_dir,  # noqa: E402
                                     relative_to_project, verified_dataset_dir)

PROTOCOL = "B (TaxPro-CL): rating >= 4, 5-core lặp user và phim, chia 70/10/20 theo user, round, random.Random(42)"
POSITIVE_THRESHOLD = 4
MIN_DEGREE = 5
MIN_USER_POSITIVES = MIN_DEGREE   # tên cũ, giữ để mã đề tài (D01, T05) không phải đổi
SPLIT_SEED = 42
RATIOS = {"train": 0.7, "validation": 0.1, "test": 0.2}


def read_ratings(path):
    """u.data: user \\t item \\t rating \\t timestamp."""
    records = []
    with Path(path).open("r", encoding="latin-1") as stream:
        for line in stream:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                records.append(tuple(int(value) for value in parts))
    return records


def deduplicate_latest(records):
    latest = {}
    for user, item, rating, timestamp in records:
        key = (user, item)
        if key not in latest or timestamp >= latest[key][1]:
            latest[key] = (rating, timestamp)
    return [(u, i, r, t) for (u, i), (r, t) in latest.items()], len(records) - len(latest)


def iterative_k_core(pairs, minimum_degree=MIN_DEGREE):
    """Lặp bỏ cặp có user hoặc phim < minimum_degree tương tác cho tới khi ổn định. Trả về (tập cặp, thống kê)."""
    current = set(pairs)
    iterations = 0
    while True:
        iterations += 1
        user_degree = collections.Counter(user for user, _ in current)
        item_degree = collections.Counter(item for _, item in current)
        kept = {(user, item) for user, item in current
                if user_degree[user] >= minimum_degree and item_degree[item] >= minimum_degree}
        if kept == current:
            break
        current = kept
    return current, {"minimum_degree": int(minimum_degree), "iterations": iterations, "pairs": len(current),
                     "users": len({u for u, _ in current}), "items": len({i for _, i in current})}


def split_counts(n):
    """Số phần tử train / validation / test của một user có n tương tác (round như TaxPro-CL; test = phần còn lại)."""
    n_train = min(round(RATIOS["train"] * n), n)
    n_validation = min(round(RATIOS["validation"] * n), n - n_train)
    return n_train, n_validation, n - n_train - n_validation


def build_split(records, seed=SPLIT_SEED, threshold=POSITIVE_THRESHOLD, min_positives=MIN_DEGREE):
    """Trả về (train, validation, test, thống kê) theo mã GỐC; mỗi phần là dict user -> list phim."""
    deduplicated, duplicates = deduplicate_latest(records)
    positives = {(user, item) for user, item, rating, _ in deduplicated if rating >= threshold}
    core, core_stats = iterative_k_core(positives, min_positives)
    by_user = collections.defaultdict(list)
    for user, item in core:
        by_user[user].append(item)
    rng = random.Random(int(seed))
    train, validation, test = {}, {}, {}
    for user in sorted(by_user):
        items = sorted(by_user[user])
        rng.shuffle(items)
        n_train, n_validation, _ = split_counts(len(items))
        train[user] = items[:n_train]
        validation[user] = items[n_train:n_train + n_validation]
        test[user] = items[n_train + n_validation:]
    users_with_positive = len({u for u, _ in positives})
    stats = {"ratings_raw": len(records), "duplicate_pairs_removed": duplicates, "positive_pairs": len(positives),
             "users_with_positive": users_with_positive, "items_with_positive": len({i for _, i in positives}),
             "kcore": core_stats, "users_kept_min_positives": core_stats["users"],
             "users_removed_min_positives": users_with_positive - core_stats["users"]}
    return train, validation, test, stats


def restrict_to_catalog(train, validation, test):
    """Danh mục = phim có >= 1 tương tác train; tách và đếm đích ngoài danh mục."""
    catalog = sorted({item for items in train.values() for item in items})
    allowed = set(catalog)
    removed = {}
    for name, part in (("validation", validation), ("test", test)):
        removed[name] = {"pairs": 0, "items": set()}
        for user in part:
            outside = [item for item in part[user] if item not in allowed]
            removed[name]["pairs"] += len(outside)
            removed[name]["items"].update(outside)
            part[user] = [item for item in part[user] if item in allowed]
    return catalog, {name: {"pairs": value["pairs"], "distinct_items": len(value["items"])} for name, value in removed.items()}


def split_text(part, user_map, item_map):
    """Nội dung tệp split: mỗi dòng `user item1 item2 ...` (mã nội bộ); bỏ user không còn đích."""
    lines = []
    for user in sorted(part):
        items = [item_map[item] for item in part[user]]
        if items:
            lines.append(" ".join(str(v) for v in [user_map[user]] + items))
    return "\n".join(lines) + "\n"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_genres(raw_directory):
    names = []
    with (Path(raw_directory) / "u.genre").open("r", encoding="latin-1") as stream:
        for line in stream:
            if "|" in line:
                name, index = line.strip().split("|")
                names.append((int(index), name))
    names = [name for _, name in sorted(names)]
    movies = {}
    with (Path(raw_directory) / "u.item").open("r", encoding="latin-1") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 5 + len(names):
                movies[int(parts[0])] = (parts[1], [int(v) for v in parts[5:5 + len(names)]])
    return names, movies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=raw_dataset_dir())
    parser.add_argument("--output", type=Path, default=verified_dataset_dir(DATASET_NAME))
    parser.add_argument("--metadata", type=Path, default=METADATA_DIR / DATASET_NAME)
    args = parser.parse_args(argv)
    if not (args.raw / "u.data").is_file():
        print("Thiếu {}. Chạy trước: python -m tools.data.download_ml100k".format(args.raw / "u.data"))
        return 1
    records = read_ratings(args.raw / "u.data")
    train, validation, test, stats = build_split(records)
    catalog, removed = restrict_to_catalog(train, validation, test)
    users = sorted(train)
    user_map = {user: index for index, user in enumerate(users)}
    item_map = {item: index for index, item in enumerate(catalog)}
    args.output.mkdir(parents=True, exist_ok=True)
    hashes, counts = {}, {}
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        text = split_text(part, user_map, item_map)
        (args.output / "{}.txt".format(name)).write_bytes(text.encode("utf-8"))
        hashes[name] = sha256_text(text)
        counts[name] = sum(len(v) for v in part.values())
    (args.output / "user_list.txt").write_bytes(
        ("original_id remap_id\n" + "".join("{} {}\n".format(u, user_map[u]) for u in users)).encode("utf-8"))
    (args.output / "item_list.txt").write_bytes(
        ("original_id remap_id\n" + "".join("{} {}\n".format(i, item_map[i]) for i in catalog)).encode("utf-8"))
    names, movies = read_genres(args.raw)
    genre_names = [name for name in names if name != "unknown"]
    args.metadata.mkdir(parents=True, exist_ok=True)
    (args.metadata / "genre_names.json").write_bytes(json.dumps(genre_names, ensure_ascii=False, indent=2).encode("utf-8"))
    no_label = 0
    with (args.metadata / "item_genres.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "original_item_id", "title"] + genre_names)
        for item in catalog:
            title, flags = movies.get(item, ("", [0] * len(names)))
            flags = [flag for name, flag in zip(names, flags) if name != "unknown"]
            no_label += int(sum(flags) == 0)
            writer.writerow([item_map[item], item, title] + flags)
    manifest = {"dataset": DATASET_NAME, "protocol": "G1", "split_protocol": PROTOCOL, "split_seed": SPLIT_SEED,
                "positive_threshold": POSITIVE_THRESHOLD, "min_user_positives": MIN_DEGREE,
                "ratios": "train round(0.7n), validation round(0.1n), test phần còn lại (giao thức B TaxPro-CL)",
                "item_kcore_filter": True, "kcore_minimum_degree": MIN_DEGREE, "rng": "python random.Random(42)",
                "statistics": stats, "users": len(users), "catalog_items": len(catalog), "interactions": counts,
                "targets_outside_catalog_removed": removed, "items_without_valid_genre": no_label,
                "split_sha256": hashes, "source": relative_to_project(args.raw / "u.data")}
    (args.output / "split_manifest.json").write_bytes(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
