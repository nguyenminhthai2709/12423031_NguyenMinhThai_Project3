# K01 – So sánh prototype ở cấp phân loại rộng và hẹp

- **Mức:** Khá · **Số sinh viên:** 1 · **Sinh viên:** Nguyễn Minh Thái (12423031, SV1)
- **Nhóm câu hỏi C:** Nên xây dựng và gộp vector đại diện nhóm như thế nào?
- **Câu hỏi nghiên cứu:** Prototype ở lá, ở cha và kết hợp hai cấp cho kết quả khác nhau thế nào khi giữ nguyên mô hình nền?
- **Phiếu đề tài trong bộ đề:** trang 43–44. Bản rút gọn kèm kế hoạch tuần: [docs/PHIEU_K01.md](docs/PHIEU_K01.md)
- **Hướng dẫn chung về codebase:** [docs/HUONG_DAN_CHUNG.md](docs/HUONG_DAN_CHUNG.md)

## Chạy theo thứ tự

```bash
python -m pip install -r requirements.txt     # Python >= 3.10
python -m tools.data.prepare
python -m pytest -q                              # test lõi phải ĐẠT; test hợp đồng hiện CHƯA LÀM
python main.py --model BPRMF --seeds 42 0 1 --tag base
python -m tools.experiments.run_K01
python -m tools.progress.check                   # tự kiểm tra tiến độ (chạy hằng tuần)
```

Thiết bị tự chọn (`--device auto`, mặc định): có GPU thì dùng GPU, không có thì CPU; ép CPU bằng `--device cpu`. Colab/Kaggle chỉ cần khi máy quá chậm (xem docs/HUONG_DAN_CHUNG.md).

**Chạy trên MovieLens 1M** (lõi 1.2): đặt biến môi trường `P3_DATASET=ml-1m` rồi chạy đúng các lệnh trên (PowerShell: `$env:P3_DATASET = "ml-1m"`; bash: `export P3_DATASET=ml-1m`). Dữ liệu vào `dataset/ml-1m/`, split vào `dataset_verify/ml-1m*/`, run nền vào `log/<mô hình>/ml-1m*/`, bảng kết quả vào `results/ml-1m/` (không ghi đè kết quả ML-100K trong `results/`). Bỏ biến này để quay lại ML-100K.

## Việc sinh viên phải cài

Các hàm/lớp dưới đây đang `raise NotImplementedError`. Cài xong, chạy `python -m tools.progress.check`: test hợp đồng tương ứng chuyển từ CHƯA LÀM sang ĐẠT (hoặc SAI nếu chưa đúng).

| Tệp | Hàm / lớp | Phụ trách |
|---|---|---|
| `models/K01_two_level.py` | `parent_membership` | SV1 |
| `models/K01_two_level.py` | `two_level_vectors` | SV1 |

Ngoài phần mã: chạy đủ seed, khóa cấu hình trước khi mở test, phân tích kết quả, vẽ hình, viết báo cáo và slide (để trong `report/`).

## Tiêu chí hoàn thành (mục 8 của phiếu) ↔ test hợp đồng

- [ ] Bảng ánh xạ 18 nhãn lá → 5 nút cha được khóa và kiểm tra (bản 1 SV; phiếu A-Small yêu cầu 20 đường phân loại đối chiếu nguồn).
- [ ] Prototype cha khớp phép tính trực tiếp từ phim con.
- [ ] Ghi số phim mất nhãn và không coi gốc duy nhất là tín hiệu phân cấp.

Kiểm tra tự động trong `tests/test_contract_K01.py`:

- `test_tc1_tree_matches_appendix` – TC1: 18 lá MovieLens (bỏ unknown), 5 cha, mỗi lá thuộc đúng một cha; tên lá đúng chuỗi trong u.genre
- `test_tc2_parent_prototype_is_direct_mean_of_children` – TC2: phim có hai lá cùng cha chỉ góp một lần; prototype cha = trung bình trực tiếp phim con phân biệt
- `test_tc2b_two_level_vectors` – TC2b: vector hỗn hợp = cân thang G2 của 0,5 lá + 0,5 cha; phim không nhãn quay về CF
- `test_tc3_coverage_and_no_single_root` – TC3: có bảng độ phủ (số phim mất nhãn, số node mỗi cấp), không dùng gốc duy nhất; có kết quả 2 seed

## Kết quả cần có trong `results/`

- `K01_validation.csv`
- `K01_test.csv`
- `K01_tree_coverage.csv`
- `K01_node_support.csv`

## Quy tắc

1. **Không sửa các tệp lõi** (danh sách trong `core_manifest.json`). Test `test_core_files_unchanged` báo SAI nếu lõi bị sửa. Gặp lỗi hoặc thiếu chức năng ở lõi: báo GV, GV sửa bản gốc và phát hành bản mới.
2. Chỉ cài vào tệp của đề tài (`models/K01_*.py`, `tools/experiments/run_K01.py`, `configure/K01.txt`, `report/`). Được thêm tệp mới (script vẽ hình, notebook phân tích).
3. Chọn cấu hình bằng validation; mở test **một lần** sau khi khóa cấu hình. Ghi nhật ký mọi thay đổi cấu hình (nếu sửa `configure/K01.txt`, ghi lý do).
4. Không phân phối lại dữ liệu MovieLens (thư mục `dataset*/`, `preprocessed/`, `metadata/ml-*/` đã được bỏ qua trong `.gitignore`).
5. Mọi bảng trong báo cáo sinh từ CSV trong `results/`, không nhập tay.
