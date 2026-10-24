# Nhật ký thực hiện K01

Ghi mỗi tuần: việc đã làm, ai làm, lệnh đã chạy, thay đổi cấu hình (và lý do), vướng mắc cần hỏi GV.

| Tuần | Ngày | Người | Việc đã làm / lệnh đã chạy | Vướng mắc |
|:---:|:---:|:---:|---|---|
| 1 | 24/10/2026 | Nguyễn Minh Thái (SV1) | - Khảo sát toàn bộ tài liệu (PHIEU_K01.md, HUONG_DAN_CHUNG.md, convention_tree.json) và 53 tệp lõi trong core_manifest.json.<br>- Kiểm tra môi trường: Python 3.13.5, PyTorch 2.13.0, NumPy 2.5.1, SciPy 1.16.3, PyTest 9.1.1 (đạt).<br>- Chuẩn bị dữ liệu: `$env:PYTHONIOENCODING="utf-8"; python -m tools.data.prepare` (Tải ml-100k.zip, kiểm tra MD5 0e33842e24a9c977be4e0107933c0723, sinh split G1 38.067/5.448/10.898).<br>- Chạy kiểm thử: `python -m pytest -q` (23 passed, 4 skipped - CHƯA LÀM).<br>- Kiểm tra tiến độ: `python -m tools.progress.check` (core.ok: true, data_ready: true).<br>- Xác định 2 hàm TODO trong models/K01_two_level.py và lập kế hoạch Tuần 2. | Gặp lỗi UnicodeEncodeError khi in tiếng Việt trên PowerShell; đã khắc phục bằng biến môi trường `$env:PYTHONIOENCODING="utf-8"` theo tài liệu GV. Không sửa đổi cấu hình. |

