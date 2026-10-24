"""K01 – So sánh prototype ở cấp phân loại rộng và hẹp (bộ đề, trang 43–44; phương án 1 SV: cây quy ước ML100K).

Cây hai cấp: 18 nhãn lá MovieLens → 5 node cha quy ước (metadata/convention_tree.json). KHÔNG dùng gốc duy nhất
làm tín hiệu phân cấp.
- Prototype lá: như G2 (trung bình phim train thuộc lá, mỗi phim góp một lần trong lá).
- Prototype cha: tính TRỰC TIẾP từ tập phim con phân biệt của cha (mỗi phim góp đúng một lần cho cha đó),
  KHÔNG lấy trung bình đều các tâm lá có kích thước khác nhau.
- Vector lá của phim: gộp đều các lá hợp lệ. Vector cha của phim: gộp đều các cha PHÂN BIỆT (không tính lặp
  khi phim có hai lá cùng cha). Hỗn hợp hai cấp: 0,5 · lá + 0,5 · cha, rồi cân thang theo G2.
"""

import numpy as np  # noqa: F401


def load_tree(path):
    """Đọc cây quy ước. Trả về {tên lá: mã cha} và danh sách mã cha theo thứ tự trong tệp."""
    import json
    from pathlib import Path
    tree = json.loads(Path(path).read_text(encoding="utf-8"))
    parent_of = {leaf: parent["id"] for parent in tree["parents"] for leaf in parent["leaves"]}
    return parent_of, [parent["id"] for parent in tree["parents"]]


def parent_membership(item_labels, genre_names, parent_of, parent_ids):
    """Ma trận bool [số phim x số cha]: phim thuộc cha nếu có ít nhất một lá thuộc cha đó (mỗi cặp một lần).

    Báo lỗi (ValueError) nếu một tên lá trong genre_names không có trong cây.
    """
    raise NotImplementedError("parent_membership")


def two_level_vectors(item_vectors, item_labels, membership, train_counts, level):
    """Vector taxonomy đã cân thang G2 cho level thuộc {"leaf", "parent", "mix"}. Trả về (vectors, has_valid).

    Gợi ý: utility.utility_function.prototype có label_prototypes, uniform_label_weights, pool_items,
    scale_like_items. "mix" = cân thang G2 của (0,5 · vector lá đã gộp + 0,5 · vector cha đã gộp).
    """
    raise NotImplementedError("two_level_vectors")
