"""Đọc nhãn thể loại phim (MovieLens 100K: 18 thể loại phẳng, đa nhãn; nhãn `unknown` đã bị bỏ).

Tệp metadata/ml-100k/item_genres.csv do tools/data/build_g1.py sinh ra, dùng mã phim nội bộ (0..I-1) giống
dataset_verify/. Một phim có thể thuộc nhiều thể loại. Phim không còn nhãn hợp lệ có hàng toàn 0.
"""

import csv
import json

import numpy as np

from config_path.config_path import DATASET_NAME, genre_names_file, item_genres_file


def load_item_genres(dataset_name=DATASET_NAME, num_items=None):
    """Trả về (labels, genre_names): labels là mảng bool [số phim x 18], hàng i = các thể loại của phim i."""
    names = json.loads(genre_names_file(dataset_name).read_text(encoding="utf-8"))
    rows = {}
    with item_genres_file(dataset_name).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rows[int(row["item_id"])] = [int(row[name]) for name in names]
    count = (max(rows) + 1) if num_items is None else int(num_items)
    labels = np.zeros((count, len(names)), dtype=bool)
    for item, flags in rows.items():
        if item < count:
            labels[item] = np.asarray(flags, dtype=bool)
    return labels, names


def load_item_titles(dataset_name=DATASET_NAME):
    titles = {}
    with item_genres_file(dataset_name).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            titles[int(row["item_id"])] = row["title"]
    return titles
