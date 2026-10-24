"""Lõi chung không bị sửa: mã băm từng tệp lõi phải khớp core_manifest.json.

Nếu cần sửa lõi (phát hiện lỗi, thiếu chức năng), báo GV; GV sửa bản gốc và phát hành phiên bản mới.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_files_unchanged():
    """Các tệp lõi chung giữ nguyên như bản GV phát hành"""
    manifest = json.loads((ROOT / "core_manifest.json").read_text(encoding="utf-8"))
    changed = [name for name, expected in manifest["files"].items()
               if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
    assert not changed, "Tệp lõi đã bị sửa hoặc mất: {}".format(changed)
