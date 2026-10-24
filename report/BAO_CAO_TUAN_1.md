# BÁO CÁO TIẾN ĐỘ TUẦN 1 – PROJECT 3
**Đề tài:** K01 – So sánh prototype ở cấp phân loại rộng và hẹp  
**Sinh viên thực hiện:** Nguyễn Minh Thái (MSSV: 12423031, SV1)  
**Thời gian thực hiện:** Tuần 1 (24/10/2026)  
**Môi trường:** Windows 11, PowerShell, Python 3.13.5  

---

## 1. Tóm tắt mục tiêu K01
- **Câu hỏi nghiên cứu:** Prototype ở lá, ở cha và kết hợp hai cấp cho kết quả khác nhau thế nào khi giữ nguyên mô hình nền BPR-MF?
- **Phạm vi phương án 1 SV:**
  - Sử dụng cây thể loại MovieLens quy ước gồm **18 nhãn lá** ánh xạ vào **5 nút cha** (khóa trong `metadata/convention_tree.json`).
  - Không sử dụng gốc duy nhất làm tín hiệu phân cấp.
  - So sánh 3 phương pháp: Collaborative Filtering gốc (`CF`), Prototype lá (`leaf`), và Hỗn hợp hai cấp (`mix` = 0.5 lá + 0.5 cha với hệ số trộn cố định $g = 0.5$).
  - Đánh giá trên 3 seed: 42, 0, 1 theo giao thức chia dữ liệu G1.

---

## 2. Các công việc Tuần 1 đã thực hiện và trạng thái
Theo kế hoạch 10 tuần quy định tại `docs/PHIEU_K01.md`:

| STT | Nhiệm vụ | Trạng thái | Ghi chú & Kết quả |
|:---:|---|:---:|---|
| 1 | Khảo sát tài liệu K01 và 53 tệp lõi trong `core_manifest.json` | **ĐÃ HOÀN THÀNH** | Đọc hiểu `PHIEU_K01.md`, `HUONG_DAN_CHUNG.md`, `convention_tree.json`. Xác định rõ các giới hạn sửa đổi code. |
| 2 | Kiểm tra môi trường Python 3.13 và các thư viện cần thiết | **ĐÃ HOÀN THÀNH** | Python 3.13.5, pip 26.2.1. Đã kiểm tra import thành công các thư viện: `torch` 2.13.0, `numpy` 2.5.1, `scipy` 1.16.3, `pytest` 9.1.1, `matplotlib` 3.10.3, `pandas` 2.2.2, `tqdm` 4.69.0. |
| 3 | Chuẩn bị dữ liệu MovieLens 100K theo giao thức G1 | **ĐÃ HOÀN THÀNH** | Chạy `$env:PYTHONIOENCODING="utf-8"; python -m tools.data.prepare`. Tải thành công `ml-100k.zip` (MD5: `0e33842e24a9c977be4e0107933c0723`). Sinh split G1 (38.067 train / 5.448 val / 10.898 test). |
| 4 | Chạy kiểm thử ban đầu của dự án | **ĐÃ HOÀN THÀNH** | Chạy `python -m pytest -q`. Kết quả: **23 passed, 4 skipped**. Toàn bộ 23 bài test lõi đều ĐẠT; 4 test hợp đồng K01 tạm bỏ qua vì hàm đang `raise NotImplementedError`. |
| 5 | Kiểm tra tiến độ tự động | **ĐÃ HOÀN THÀNH** | Chạy `python -m tools.progress.check`. Kết quả: `core.ok: true`, `data_ready: true`. Lưu trạng thái vào `results/progress_status.json`. |
| 6 | Xác định các phần còn TODO cho K01 | **ĐÃ HOÀN THÀNH** | Xác định 2 hàm cần cài đặt trong `models/K01_two_level.py`: `parent_membership` và `two_level_vectors`. |
| 7 | Ghi nhật ký nghiên cứu | **ĐÃ HOÀN THÀNH** | Cập nhật `report/NHAT_KY.md` đầy đủ nội dung, lệnh và cách xử lý lỗi. |

---

## 3. Môi trường và Dữ liệu thực tế
- **Đường dẫn repo:** `C:\Project3\K01_NguyenMinhThai-20261023T230635Z-1-001\K01_NguyenMinhThai`
- **Dữ liệu MovieLens 100K (Giao thức G1, seed 42):**
  - Số người dùng (users): 938
  - Số phim trong danh mục (items): 1.008
  - Tập Train: 38.067 tương tác dương
  - Tập Validation: 5.448 tương tác dương
  - Tập Test: 10.898 tương tác dương
  - Phân nhóm tương tác:
    - `very_few_1_5`: 131 phim
    - `long_tail_1_10`: 321 phim
    - `warm_gt10`: 687 phim

---

## 4. Kết quả kiểm thử và tiến độ
- **`pytest`**:
  ```text
  ssss.......................                                              [100%]
  23 passed, 4 skipped, 1 warning in 8.40s
  ```
- **`tools.progress.check`**:
  ```text
  Đề tài: K01 - So sánh prototype ở cấp phân loại rộng và hẹp
  Dữ liệu ml-100k đã chuẩn bị: có
  Lõi chung nguyên vẹn: có
  Run hoàn tất: chưa có
  Tệp trong results/: .gitkeep
  Tổng hợp: {'hợp đồng CHƯA LÀM': 4, 'lõi ĐẠT': 23}
  ```

---

## 5. Những phần code đã sửa và lý do
- Trong Tuần 1, **chỉ cập nhật tệp `report/NHAT_KY.md`** để ghi lại nhật ký công việc.
- **Không có bất kỳ tệp lõi hay tệp cấu hình nào bị thay đổi**. Tệp `tests/test_core_intact.py` xác nhận 53 tệp lõi hoàn toàn nguyên vẹn so với mã băm trong `core_manifest.json`.

---

## 6. Lỗi hoặc khó khăn gặp phải và giải pháp
- **Lỗi gặp phải:** Lệnh `python -m tools.data.prepare` ban đầu bị ngắt bởi ngoại lệ `UnicodeEncodeError: 'charmap' codec can't encode character '\u0110'` do PowerShell mặc định sử dụng bảng mã cp1252 khi in ký tự tiếng Việt (`Đang tải`).
- **Giải pháp:** Thiết lập biến môi trường `$env:PYTHONIOENCODING="utf-8"` trước khi gọi lệnh Python. Lệnh chạy lại thành công 100%.

---

## 7. Minh chứng lưu trữ
1. Tệp split và manifest: `dataset_verify/ml-100k/split_manifest.json`
2. Trạng thái tiến độ: `results/progress_status.json`
3. Nhật ký thực hiện: `report/NHAT_KY.md`
4. Báo cáo tuần 1: `report/BAO_CAO_TUAN_1.md`

---

## 8. Kế hoạch Tuần 2
1. Khóa và kiểm tra cấu trúc cây quy ước trong `metadata/convention_tree.json` (18 lá -> 5 cha).
2. Triển khai hàm `parent_membership(item_labels, genre_names, parent_of, parent_ids)` trong `models/K01_two_level.py`:
   - Xây dựng ma trận boolean kích thước `[num_items, num_parents]`.
   - Vượt qua bài kiểm tra hợp đồng TC1 (`test_tc1_tree_matches_appendix`) và bước đầu của TC2 (`test_tc2_parent_prototype_is_direct_mean_of_children`).
3. Chuẩn bị logic tính toán cho hàm `two_level_vectors`.
