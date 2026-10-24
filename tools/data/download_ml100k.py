"""Tải MovieLens (mặc định 100K; 1M khi đặt P3_DATASET=ml-1m) từ GroupLens và kiểm tra checksum (MD5 chính thức + SHA-256).

Điều khoản của GroupLens không cho phân phối lại dữ liệu, nên mỗi nhóm tự tải bằng script này.
Nếu máy không tải được (lỗi chứng chỉ SSL, mạng chặn): tải thủ công tệp <dataset>.zip từ
https://files.grouplens.org/datasets/movielens/ bằng trình duyệt, đặt vào thư mục dataset/
rồi chạy lại lệnh; script sẽ chỉ kiểm tra checksum và giải nén.

ML-1M (ratings.dat, movies.dat, phân cách "::", thể loại ghi bằng tên) được CHUẨN HÓA sang đúng định dạng ML-100K
ngay sau khi giải nén, để mọi bước sau (build_g1, các đề tài) dùng chung một bộ đọc:
    u.data  : user \\t item \\t rating \\t timestamp   (giữ nguyên mọi dòng của ratings.dat, không lọc)
    u.genre : 19 dòng "tên|chỉ số" theo đúng thứ tự u.genre của ML-100K (unknown = 0)
    u.item  : item|tiêu đề|||| + 19 cờ 0/1   (ngày phát hành và URL không có trong ML-1M → để trống)
Tệp gốc giữ nguyên; normalization_manifest.json ghi số dòng và mã băm hai phía.

    python -m tools.data.download_ml100k
"""

import hashlib
import json
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import DATASET_DIR, DATASET_NAME, MOVIELENS  # noqa: E402

ML100K_GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
                 "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def digest(path, algorithm):
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _open(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Project3 data download)"})
    try:
        return urllib.request.urlopen(request, timeout=120)
    except urllib.error.URLError as error:
        if not isinstance(getattr(error, "reason", None), ssl.SSLError):
            raise
        try:  # một số máy Windows thiếu chứng chỉ gốc: thử bộ chứng chỉ của certifi nếu có
            import certifi
        except ImportError:
            raise error
        return urllib.request.urlopen(request, timeout=120, context=ssl.create_default_context(cafile=certifi.where()))


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    with _open(url) as response, temporary.open("wb") as stream:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            stream.write(chunk)
    temporary.replace(target)


def normalize_ml1m(directory):
    """ratings.dat/movies.dat → u.data/u.genre/u.item (định dạng ML-100K). Không lọc, không sửa giá trị."""
    directory = Path(directory)
    ratings = (directory / "ratings.dat").read_bytes().decode("latin-1").splitlines()
    movies = (directory / "movies.dat").read_bytes().decode("latin-1").splitlines()
    data_lines = []
    for number, line in enumerate(ratings, start=1):
        parts = line.split("::")
        if len(parts) != 4:
            raise ValueError("ratings.dat dòng {} sai định dạng: {!r}".format(number, line))
        data_lines.append("\t".join(parts))
    index = {name: position for position, name in enumerate(ML100K_GENRES)}
    item_lines = []
    for number, line in enumerate(movies, start=1):
        parts = line.split("::")
        if len(parts) != 3 or "|" in parts[1]:
            raise ValueError("movies.dat dòng {} không chuẩn hóa được: {!r}".format(number, line))
        flags = [0] * len(ML100K_GENRES)
        for genre in filter(None, parts[2].split("|")):
            if genre not in index:
                raise ValueError("movies.dat dòng {}: thể loại {!r} không có trong từ điển ML-100K".format(number, genre))
            flags[index[genre]] = 1
        item_lines.append("|".join([parts[0], parts[1], "", "", ""] + [str(f) for f in flags]))
    outputs = {"u.data": "\n".join(data_lines) + "\n",
               "u.genre": "".join("{}|{}\n".format(name, i) for i, name in enumerate(ML100K_GENRES)),
               "u.item": "\n".join(item_lines) + "\n"}
    for name, text in outputs.items():
        (directory / name).write_bytes(text.encode("latin-1"))
    manifest = {"source": {"ratings.dat": {"lines": len(ratings), "sha256": digest(directory / "ratings.dat", "sha256")},
                           "movies.dat": {"lines": len(movies), "sha256": digest(directory / "movies.dat", "sha256")}},
                "normalized": {name: {"lines": text.count("\n"), "sha256": digest(directory / name, "sha256")}
                               for name, text in outputs.items()},
                "rule": "u.data giữ mọi dòng ratings.dat; u.item: 18 thể loại theo tên → cờ theo thứ tự u.genre ML-100K, "
                        "unknown = 0; ngày phát hành/URL để trống"}
    (directory / "normalization_manifest.json").write_bytes(
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    return manifest


def main():
    source = MOVIELENS[DATASET_NAME]
    archive = DATASET_DIR / "{}.zip".format(DATASET_NAME)
    if not archive.is_file():
        print("Đang tải", source["url"])
        try:
            download(source["url"], archive)
        except Exception as error:  # noqa: BLE001
            print("Không tải được ({}). Hãy tải thủ công tệp {} vào {} rồi chạy lại.".format(error, archive.name, DATASET_DIR))
            return 1
    md5, sha256 = digest(archive, "md5"), digest(archive, "sha256")
    if md5 != source["md5"] or sha256 != source["sha256"]:
        print("Checksum KHÔNG khớp: md5={} sha256={}. Xóa {} và tải lại.".format(md5, sha256, archive))
        return 1
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(DATASET_DIR)
    print("Đã kiểm tra checksum (md5={}) và giải nén vào {}".format(md5, DATASET_DIR / DATASET_NAME))
    if DATASET_NAME == "ml-1m":
        manifest = normalize_ml1m(DATASET_DIR / DATASET_NAME)
        print("Đã chuẩn hóa ML-1M sang định dạng ML-100K:",
              {name: value["lines"] for name, value in manifest["normalized"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
