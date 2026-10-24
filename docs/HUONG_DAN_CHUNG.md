# Hướng dẫn chung về codebase Project 3

Mọi đề tài dùng **cùng một lõi**: cùng cách chia dữ liệu (giao thức G1), cùng bộ đánh giá theo nhóm, cùng mô hình nền và cùng cách xây prototype (G2). Nhờ vậy kết quả của các đề tài so sánh được với nhau, và giảng viên kiểm tra tiến độ của mọi nhóm bằng cùng một lệnh. Phần riêng của từng đề tài nằm trong các tệp mang mã đề tài.

## 1. Cấu trúc thư mục

| Thư mục / tệp | Nội dung | Ai sửa |
|---|---|---|
| `config_path/` | Đường dẫn chuẩn. Mọi script lấy đường dẫn từ đây, không tự ghép chuỗi | Lõi, không sửa |
| `configure/` | Cấu hình mô hình nền (`BPRMF.txt`, `LightGCN.txt`) và cấu hình thí nghiệm của đề tài (`<MÃ>.txt`) | Mô hình nền: không sửa. `<MÃ>.txt`: sửa được nhưng phải ghi lý do vào nhật ký |
| `models/` | Mô hình nền do GV cấp, cùng tệp của đề tài (`<MÃ>_*.py`) | Chỉ sửa tệp của đề tài |
| `utility/` | Đọc dữ liệu, lấy mẫu âm, đánh giá, huấn luyện, prototype G2 | Lõi, không sửa |
| `tools/data/` | Tải MovieLens 100K hoặc 1M, kiểm tra checksum, dựng split G1 | Lõi (riêng T05 có `build_c1.py` của đề tài) |
| `tools/protocol/` | Sinh nhóm phim theo số tương tác train (1–5, 1–10, > 10) | Lõi |
| `tools/experiments/` | `common.py` (lõi) và `run_<MÃ>.py` (thí nghiệm của đề tài) | Sửa `run_<MÃ>.py` khi cần |
| `tools/progress/check.py` | Tự kiểm tra tiến độ | Lõi |
| `tests/` | Test lõi (`test_core_*.py`) và test hợp đồng của đề tài (`test_contract_<MÃ>.py`) | Không sửa test có sẵn, được thêm test mới |
| `dataset/` | Dữ liệu gốc tải về | Sinh tự động, không chia sẻ lại |
| `dataset_verify/` | Split đã dựng: `train.txt`, `validation.txt`, `test.txt` | Sinh tự động |
| `preprocessed/` | Nhóm phim và tập đích theo nhóm | Sinh tự động |
| `metadata/` | Thể loại phim (sinh tự động), cây thể loại quy ước (K01) | Không sửa |
| `log/` | Mỗi lần huấn luyện tạo `log/<mô hình>/<dữ liệu>/<tag>/seed<k>/` | Sinh tự động |
| `results/` | Bảng kết quả CSV của đề tài, `progress_status.json` | Sinh tự động |
| `report/` | Báo cáo, slide, hình, `NHAT_KY.md` | Sinh viên |
| `docs/` | Phiếu đề tài rút gọn, hướng dẫn này | GV |
| `topic.json`, `core_manifest.json` | Thông tin đề tài, mã băm tệp lõi | GV, không sửa |

## 2. Chuẩn bị môi trường và dữ liệu

```bash
python -m pip install -r requirements.txt
python -m tools.data.prepare
python -m pytest -q
```

`prepare` làm ba việc:
1. Tải `ml-100k.zip` từ GroupLens và kiểm tra checksum MD5 và SHA-256.
2. Dựng split G1 vào `dataset_verify/ml-100k/`.
3. Sinh giao thức nhóm vào `preprocessed/evaluation_protocol/ml-100k/`.

Nếu máy không tải được (lỗi chứng chỉ, mạng chặn):
1. Tải tệp bằng trình duyệt từ <https://files.grouplens.org/datasets/movielens/ml-100k.zip>.
2. Đặt tệp vào `dataset/`.
3. Chạy lại lệnh. Script chỉ kiểm tra checksum rồi giải nén.

Điều khoản của GroupLens không cho phân phối lại dữ liệu, nên mỗi nhóm tự tải. Không gửi thư mục `dataset*/` cho nhóm khác.

Quy ước chia dữ liệu (lõi 1.1): dữ liệu đã có split công khai (Amazon-Book của LightGCN, dùng trong NCKH) theo
**giao thức A** của TaxPro-CL – giữ test công khai, lọc 5-core train, tách 10% validation theo user; dữ liệu chưa có
split (MovieLens 100K của 20 đề tài) theo **giao thức B** của TaxPro-CL. Split G1 của MovieLens 100K gồm các bước:
1. Coi rating ≥ 4 là tương tác dương.
2. Khử trùng cặp user–phim, giữ bản ghi muộn nhất.
3. Lọc 5-core lặp trên cả user và phim (mỗi user, mỗi phim có ≥ 5 tương tác dương).
4. Mỗi user (theo mã tăng dần, một bộ sinh Python `random.Random(42)`): sắp phim tăng dần, trộn, rồi chia round(0,7n)
   cho train, round(0,1n) cho validation, phần còn lại cho test.
5. Danh mục chỉ gồm phim có tương tác train. Đích ngoài danh mục (nếu có) được tách ra và đếm trong `split_manifest.json`.

Kết quả (giao thức B, seed 42):

| Dữ liệu | User | Phim (danh mục) | Train / validation / test |
|---|---|---|---|
| MovieLens 100K | 938 | 1.008 | 38.067 / 5.448 / 10.898 |
| MovieLens 1M | 6.034 | 3.124 | 401.932 / 57.459 / 114.980 |

### 2b. MovieLens 1M (lõi 1.2)

Mọi đề tài chạy được trên MovieLens 1M bằng đúng các lệnh của ML-100K, chỉ cần đặt biến môi trường `P3_DATASET`:

```bash
export P3_DATASET=ml-1m            # PowerShell: $env:P3_DATASET = "ml-1m"
python -m tools.data.prepare        # tải ml-1m.zip (5,9 MB), kiểm checksum, chuẩn hóa, dựng split, sinh nhóm
python main.py --model BPRMF --seeds 42 0 1 --tag base
python -m tools.experiments.run_<MÃ>
python -m pytest -q
```

- Checksum: MD5 `c4d9eecfca2ab87c1945afe126590906` (khớp `ml-1m.zip.md5` của GroupLens), SHA-256 ghi trong `config_path`.
- ML-1M có định dạng khác (`ratings.dat`, `movies.dat`, phân cách `::`, thể loại ghi bằng tên). `download_ml100k`
  chuẩn hóa sang đúng định dạng ML-100K (`u.data`, `u.item`, `u.genre`; 18 thể loại trùng tên ML-100K, `unknown` = 0)
  mà không lọc dòng nào; số dòng và mã băm hai phía ghi trong `dataset/ml-1m/normalization_manifest.json`.
- Tệp sinh ra tách riêng theo dữ liệu: `dataset_verify/ml-1m*/`, `preprocessed/evaluation_protocol/ml-1m*/`,
  `metadata/ml-1m*/`, `log/<mô hình>/ml-1m*/`, và **`results/ml-1m/`** (kết quả ML-100K vẫn ở `results/`). T05 dùng
  `ml-1m-c1`. Cấu hình ghi `dataset = ml-100k` (hoặc `ml-100k-c1`) được tự hiểu là dataset đang chọn.
- ML-1M lớn gấp ~10 lần ML-100K: huấn luyện trên CPU lâu hơn tương ứng (đo trên máy GV ghi trong báo cáo kiểm thử).

### 2c. Nhóm phim rất ít tương tác: 1–5 thay cho 1–2 (lõi 1.2)

**Lý do.** Giao thức B lọc 5-core lặp trên cả user và phim, nên phim nào còn lại cũng có ≥ 5 tương tác dương trước khi
chia; rất ít phim rơi xuống 1–2 tương tác train. Nhóm 1–2 của bộ đề gần như biến mất: trên
ML-100K chỉ còn 11 phim với 27 đích test (25 user), dưới ngưỡng thăm dò (30 user, 100 đích) nên mọi chỉ số của nhóm này
đều chỉ mang tính thăm dò và rất nhiễu. Nhóm 1–5 là nhóm nhỏ nhất có đủ đích ở cả validation và test (ML-100K: 205 đích
test / 143 user, 112 đích validation / 97 user), đồng thời trùng ngưỡng Near-Cold (NC, 1–5) của NCKH và TaxPro-CL, nên
hai phần dùng chung một định nghĩa "rất ít tương tác". Nhóm 3–10 của D01/T01 đổi tương ứng thành 6–10 để ba nhóm
1–5 / 6–10 / > 10 vẫn rời nhau và phủ hết danh mục. Bộ đề (trang 8–9, phiếu D01, T01, T05) ghi 1–2; đây là thay đổi có
chủ đích, GV/GVHD cập nhật bộ đề tương ứng.

Số phim và số đích theo ngưỡng (số tương tác train, split giao thức B):

| Nhóm (tương tác train) | ML-100K: số phim | % danh mục | Đích test / user | Đích validation / user | ML-1M: số phim | % danh mục | Đích test / user | Đích validation / user |
|---|---|---|---|---|---|---|---|---|
| 1–2 (trước lõi 1.2) | 11 | 1,1% | 27 / 25 | 12 / 12 | 11 | 0,4% | 27 / 27 | 19 / 19 |
| 1–3 | 37 | 3,7% | 76 / 63 | 37 / 34 | 48 | 1,5% | 87 / 83 | 58 / 57 |
| 1–4 | 79 | 7,8% | 139 / 110 | 70 / 62 | 128 | 4,1% | 225 / 190 | 114 / 103 |
| **1–5 (từ lõi 1.2)** | **131** | **13,0%** | **205 / 143** | **112 / 97** | **229** | **7,3%** | **389 / 304** | **195 / 169** |
| 6–10 | 190 | 18,8% | 423 / 275 | 196 / 149 | 343 | 11,0% | 801 / 576 | 415 / 329 |
| 1–10 (long-tail) | 321 | 31,8% | 628 / 337 | 308 / 211 | 572 | 18,3% | 1.190 / 756 | 610 / 457 |
| > 10 | 687 | 68,2% | 10.270 / 936 | 5.140 / 922 | 2.552 | 81,7% | 113.790 / 6.031 | 56.849 / 6.022 |

Nhóm 1–5 vẫn nhỏ: khi báo cáo, luôn ghi kèm số đích/số user (cột `targets`, `users` trong CSV) và cờ `exploratory`.

Sau khi dựng, `split_manifest.json` ghi mã băm của split. Mọi nhóm dùng cùng split, nên mã băm phải giống nhau.

## 3. Huấn luyện mô hình nền

```bash
python main.py --model BPRMF --seeds 42 0 1 --tag base
python main.py --model LightGCN --seeds 42 0 1 --tag base
python main.py --model BPRMF --seed 42 --reg_lambda 0.001 --tag l2_1e-3   # đổi một khóa cấu hình
```

- **Thiết bị:** `main.py` mặc định `--device auto` – có GPU (CUDA) thì dùng GPU, không có thì CPU; ép bằng `--device cpu` hoặc `--device cuda`. D06 đọc `device` trong `configure/D06.txt` (mặc định `auto`, ghi đè bằng biến môi trường `P3_DEVICE`). Thí nghiệm sau huấn luyện của đề tài (prototype, cổng, nhánh học thêm) chạy trên CPU vì làm việc với vector đóng băng dạng numpy. Kết quả GPU và CPU lệch rất nhỏ (đo trên ML-1M: chỉ số tổng hợp ≤ 5,5e-5).
- Mọi khóa trong `configure/<Mô hình>.txt` đều ghi đè được bằng `--<khóa> <giá trị>`. Lần chạy có cấu hình khác cần `--tag` khác.
- `num_worker` (mặc định 4, có trong cấu hình mô hình nền và cấu hình đề tài) là số luồng lấy mẫu âm song song. Máy yếu có thể giảm (ví dụ `--num_worker 2`), nhưng số luồng thay đổi cách chia lô lấy mẫu nên kết quả lệch nhẹ: muốn tái lập đúng từng số thì giữ cùng `num_worker` và ghi vào nhật ký.
- Seed huấn luyện (lõi 1.4): mọi đề tài chạy **3 seed: 42, 0, 1** (mặc định của `main.py`; khóa `seeds`/`final_seeds` trong `configure/<MÃ>.txt`). Bộ đề ghi 2026, 2027, 2028 (mức Dễ 2 seed); đổi theo yêu cầu ngày 16/10/2026 để mọi đề tài có độ lệch chuẩn trên 3 seed. Seed phụ (chọn phim kiểm tay, mặt nạ nhãn, bộ nhớ, bootstrap) và seed chia dữ liệu 42 không đổi.
- Dừng sớm (lõi 1.3, áp cho mọi lần huấn luyện mô hình nền BPR-MF và LightGCN bằng `main.py`): trần `training_epochs = 1000`, kiểm tra validation mỗi `interval = 10` epoch, dừng sau `early_stopping = 10` lần kiểm tra không cải thiện (tức 100 epoch). Bộ đề ghi 100 / 5 / 4; đổi theo yêu cầu ngày 16/10/2026. Thí nghiệm học thêm số epoch cố định, không dừng sớm (K03 `extra_epochs`, K04 `branch_epochs`), giữ nguyên.
- Validation được chạy mỗi `interval` epoch. Checkpoint tốt nhất theo NDCG@10 validation toàn bộ được lưu lại. Huấn luyện dừng sớm sau `early_stopping` lần không cải thiện. Sau đó test được mở **một lần** với checkpoint tốt nhất.
- Mỗi seed tạo các tệp sau:
  - `config_resolved.json` (cấu hình thực dùng) và `environment.json` (phiên bản Python, thư viện, thiết bị);
  - `run_manifest.json` (trạng thái `completed` khi chạy xong) và `training.log`;
  - `validation_metrics.json`, `best_validation_model.pt`, `last_model.pt`;
  - `final_test_metrics.json`, `final_test_groups.csv`, `final_test_per_user.csv`.
- Ghi thời gian thực tế mỗi seed vào nhật ký (`training.log` có sẵn thời gian). Máy chậm thì chạy trên Google Colab hoặc Kaggle: chép cả thư mục đề tài lên và chạy cùng lệnh. Muốn để `log/` ở nơi khác (ví dụ Google Drive), đặt biến môi trường `P3_LOG_ROOT`.

## 4. Dùng lại checkpoint cho thí nghiệm của đề tài

Phần lớn đề tài làm việc trên **vector đóng băng** của mô hình nền (prototype, bộ trộn, cổng):

```python
from tools.experiments import common
from utility.utility_train import posthoc
bundle = common.load_base("LightGCN", seed=42, tag="base")   # nạp checkpoint tốt nhất
U, V = posthoc.frozen_embeddings(bundle)                         # numpy: vector user, vector phim
result, definitions = common.evaluate(bundle.dataset, config, common.dot_scores(U, V_moi), "validation")
```

Mọi bộ chấm điểm phải đi qua `common.evaluate`. Nhờ vậy các phương pháp có cùng ứng viên, cùng mặt nạ, cùng quy tắc phá hòa và cùng cách tính chỉ số. **Không tự viết lại hàm tính Recall/NDCG.**

## 5. Đánh giá

- **Xếp hạng toàn danh mục.** Validation loại dương train. Test loại thêm dương validation, nhưng không loại đích test. Phim bị loại nhận điểm −∞.
- **Phá hòa.** Khi điểm bằng nhau, thứ tự được quyết định bằng một hoán vị mã phim cố định (`tiebreak_seed`), giống nhau ở mọi phương pháp.
- **Chỉ số.** Recall@10 và NDCG@10 là bắt buộc, @20 là tùy chọn. Chỉ số nhóm giữ nguyên bảng xếp hạng đầy đủ, chỉ thay tập đích:
  - `very_few_1_5`: phim có 1–5 tương tác train (lõi 1.2; trước là `very_few_1_2` – lý do ở mục 2c);
  - `long_tail_1_10`: phim có 1–10 tương tác train;
  - `warm_gt10`: phim có hơn 10 tương tác train.
- **Nhãn thăm dò.** Nhóm có dưới 30 user hoặc dưới 100 đích được gắn `exploratory = 1`; kết luận về nhóm đó chỉ mang tính thăm dò. Không lấy trung bình các nhóm để suy ra chỉ số toàn bộ.
- **Chọn cấu hình.** Chọn trên validation, khóa lại, rồi mới mở test. Không chọn theo từng nhóm test.

## 6. Tự kiểm tra tiến độ

```bash
python -m tools.progress.check
```

Lệnh này in ra:
- dữ liệu đã chuẩn bị chưa;
- lõi chung còn nguyên vẹn không;
- các run đã hoàn tất;
- các tệp trong `results/`;
- trạng thái từng test.

Kết quả đồng thời được ghi vào `results/progress_status.json`.

| Trạng thái | Nghĩa |
|---|---|
| ĐẠT | Test qua |
| CHƯA LÀM | Hàm còn để TODO, hoặc chưa có tệp kết quả cần kiểm tra |
| SAI | Đã làm nhưng chưa đúng tiêu chí. Đọc thông báo lỗi bên cạnh |
| KHÔNG ÁP DỤNG | Test lõi của thành phần mà đề tài không dùng |

GV chạy cùng lệnh này hàng loạt cho mọi thư mục, nên đừng đổi tên tệp kết quả hay tên cột đã quy định.

## 7. Quy tắc

1. **Không sửa tệp lõi.** Danh sách tệp lõi nằm trong `core_manifest.json`. Nếu phát hiện lỗi ở lõi, báo GV kèm lệnh tái hiện lỗi; GV sửa bản gốc và phát hành bản mới cho mọi nhóm.
2. Mở test một lần sau khi khóa cấu hình. Mọi lần đổi cấu hình phải ghi vào `report/NHAT_KY.md`.
3. Mọi số trong báo cáo sinh từ CSV trong `results/`. Hình vẽ cũng sinh từ CSV bằng script để trong `report/`.
4. Kết quả không cải thiện vẫn được đánh giá đầy đủ nếu làm đúng phương pháp. Không tìm split, seed hay cấu hình "thuận lợi" sau khi đã xem test.
5. Ghi nguồn khi dùng mã, checkpoint hoặc dữ liệu của người khác (kể cả checkpoint tham chiếu do GV cấp).

## 8. Gặp lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Thiếu train ... Chạy trước: python -m tools.data.prepare` | Chưa chuẩn bị dữ liệu |
| `Chưa có run hoàn tất của ...` khi chạy `run_<MÃ>.py` | Chưa huấn luyện mô hình nền với đúng `--tag` và seed trong `configure/<MÃ>.txt` |
| Lỗi in tiếng Việt trên cửa sổ lệnh Windows | Chạy `set PYTHONIOENCODING=utf-8` (cmd) hoặc `$env:PYTHONIOENCODING="utf-8"` (PowerShell) |
| Lỗi chứng chỉ SSL khi tải dữ liệu | Tải thủ công như mục 2 |
| `test_core_files_unchanged` SAI | Một tệp lõi đã bị sửa. Khôi phục tệp đó từ bản GV phát |
