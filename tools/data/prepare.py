"""Chuẩn bị dữ liệu trong một lệnh: tải ML100K + kiểm tra checksum → dựng split G1 → sinh giao thức nhóm.

    python -m tools.data.prepare
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data import build_g1, download_ml100k  # noqa: E402
from tools.protocol import build_protocol  # noqa: E402


def main():
    if download_ml100k.main() != 0:
        return 1
    if build_g1.main([]) != 0:
        return 1
    return build_protocol.main([])


if __name__ == "__main__":
    sys.exit(main())
