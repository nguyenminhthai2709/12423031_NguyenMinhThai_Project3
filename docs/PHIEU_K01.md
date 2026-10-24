# Phiếu đề tài K01 – So sánh prototype ở cấp phân loại rộng và hẹp

Bản rút gọn từ bộ đề (trang 43–44), kèm cách thực hiện trên codebase này. Khi có khác biệt, bộ đề và hướng dẫn của GV là căn cứ.

| Mục | Nội dung |
|---|---|
| Mức | Khá |
| Sinh viên | Nguyễn Minh Thái (SV1) |
| Trọng tâm | Hai cấp phân loại |
| Seed | 3 seed nền BPR-MF (42, 0, 1) |
| Độ dài báo cáo | 10–12 trang (không tính phụ lục) |

## 1. Động cơ

Phân loại rộng có nhiều phim hỗ trợ nhưng ít chi tiết; phân loại hẹp cụ thể hơn nhưng có thể thiếu dữ liệu. Sinh viên kiểm tra sự đánh đổi này trên một cây nhỏ thay vì khung đa cấp đầy đủ.

## 2. Câu hỏi nghiên cứu

Prototype ở lá, ở cha và kết hợp hai cấp cho kết quả khác nhau thế nào khi giữ nguyên mô hình nền?

## 3. Dữ liệu và kiến thức cần bổ sung

ML100K + cây quy ước (phụ lục trang 55); gói A-Small (Amazon Reviews 2018, Software) chỉ là mở rộng có điều kiện. Cần học thêm: quan hệ cha–con, đường phân loại, khử trùng phim theo node.

## 4. Cách làm tối thiểu

Bản 1 SV: (1) dùng cây thể loại MovieLens quy ước ở phụ lục trang 55 (18 nhãn lá → 5 nút cha; bỏ nhãn unknown). (2) Dùng BPR-MF; xây prototype lá và hỗn hợp hai cấp (một nửa mỗi cấp) trên vector đóng băng; prototype cha tính trực tiếp từ tập phim con phân biệt. (3) g = 0,5 cho các biến thể, thêm CF gốc; khóa bảng ánh xạ trước khi xem kết quả.

## 5. Thí nghiệm và đối chứng

G1–G2 trên ML100K với cây quy ước; bản 1 SV: 3 seed nền, 3 phương pháp (CF, lá, hỗn hợp). Recall@10, NDCG@10 theo nhóm hỗ trợ và số node mỗi cấp.

## 6. Phân công

1 SV đảm nhận toàn bộ (phiếu 2 SV: SV1 kiểm tra cây và dựng bộ nhớ hai cấp; SV2 chạy BPR-MF và đối chứng).

**Phạm vi theo số sinh viên:** Đề tài 1 SV → theo phương án của bộ đề: dùng cây MovieLens quy ước; so lá và hỗn hợp, cộng CF; 3 seed (bộ đề ghi 2 seed; lõi 1.4 dùng 42, 0, 1).

## 7. Ba mốc kiểm tra

| Mốc | Cần trình bày |
|---|---|
| Cuối tuần 3 | Split G1; bảng ánh xạ cây quy ước đã khóa; thống kê cây; BPR-MF nền 3 seed. |
| Cuối tuần 6 | Bảng validation CF / lá / hỗn hợp; kiểm thử prototype cha tính trực tiếp từ phim con. |
| Cuối tuần 10 | Bảng test theo nhóm hỗ trợ, phân tích theo kích thước node; báo cáo 10–12 trang. |

## 8. Tiêu chí hoàn thành

- [ ] Bảng ánh xạ 18 nhãn lá → 5 nút cha được khóa và kiểm tra (bản 1 SV; phiếu A-Small yêu cầu 20 đường phân loại đối chiếu nguồn).
- [ ] Prototype cha khớp phép tính trực tiếp từ phim con.
- [ ] Ghi số phim mất nhãn và không coi gốc duy nhất là tín hiệu phân cấp.

## 9. Nguồn lực và rủi ro

GPU tùy chọn khoảng 5–12 giờ (ước lượng cho bản đầy đủ). Công cụ: PyTorch, numpy, pandas, matplotlib.

Cây quy ước không phải cây chính thức của MovieLens: chỉ kết luận về cấp độ gộp dưới ánh xạ đã công bố; không xác nhận taxonomy tự nhiên.

## 10. Liên hệ với hướng nghiên cứu chung và điểm dừng

Tiếp cận ý tưởng prototype đa cấp; không thêm attention, sibling loss hay SSL đa cấp.

Dừng khi trả lời đủ câu hỏi và đạt tiêu chí trên. Không bắt buộc xây ứng dụng web, thêm bộ dữ liệu hoặc thêm loss. Mở rộng chỉ sau khi nghiệm thu phần bắt buộc.

## 11. Mở rộng (chỉ làm khi xong phần bắt buộc)

- **Biến thể chỉ dùng prototype cha; chạy 3 seed (phạm vi phiếu 2 SV)** – Chỉ làm sau khi nghiệm thu phần bắt buộc.
- **Thử trên gói A-Small (Amazon Reviews 2018, Software) nếu GV đã chuẩn bị** – Gói gồm interactions.csv, item_paths.json, README nguồn; tối đa 100.000 tương tác dương; chỉ làm khi đủ điều kiện giao và còn thời gian.

## 12. Kế hoạch 10 tuần

Split G1, bộ đánh giá theo nhóm và mô hình nền đã có trong **lõi chung**. Ở tuần 1–2, việc "cài split/bộ đánh giá" được thay bằng: đọc hiểu mã lõi, chạy `python -m pytest -q`, tự tính tay ví dụ trang 9 và ví dụ riêng của đề tài rồi đối chiếu với mã. Phần riêng của đề tài vẫn do sinh viên cài.

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 1 | Đọc phiếu K01, phụ lục cây; 3 bài phân cấp; cài môi trường | Ghi chép đọc; repo khung |
| 2 | Split G1, bộ đánh giá; nạp và khóa cây quy ước | Ánh xạ lá → cha |
| 3 | BPR-MF nền 3 seed; thống kê cây → Mốc 1 | Checkpoint; bảng cây |
| 4 | Prototype lá và cha (tính trực tiếp) | Bộ nhớ hai cấp |
| 5 | Vector lá / hỗn hợp; bộ chấm điểm g = 0,5 | Mô-đun + kiểm thử |
| 6 | Bảng validation → Mốc 2 | Bảng validation |
| 7 | Khóa; 3 seed cuối | Nhật ký cấu hình |
| 8 | Mở test; phân tích theo kích thước node | Bảng + biểu đồ |
| 9 | Kiểm tra tái lập; bản nháp 10–12 trang | Bản nháp báo cáo |
| 10 | Hoàn thiện báo cáo, slide; nộp → Mốc 3 | Gói nộp |

## 13. Tài liệu tham khảo (đã kiểm chứng đường dẫn ngày 09/10/2026)

- [BB18] C.-N. Ziegler, G. Lausen, L. Schmidt-Thieme. Taxonomy-driven Computation of Product Recommendations. CIKM 2004, pp. 406–415. doi:10.1145/1031171.1031252. – Dùng cây phân loại sản phẩm để xây hồ sơ và lan sở thích lên nút cha. <https://doi.org/10.1145/1031171.1031252>
- [BB20] N. Koenigstein, G. Dror, Y. Koren. Yahoo! Music Recommendations: Modeling Music Ratings with Temporal Dynamics and Item Taxonomy. RecSys 2011, pp. 165–172. doi:10.1145/2043932.2043964. – Đưa phân cấp item (bài hát → album → nghệ sĩ → thể loại) vào mô hình phân rã. <https://doi.org/10.1145/2043932.2043964>
- [BB19] A. K. Menon, K.-P. Chitrapura, S. Garg, D. Agarwal, N. Kota. Response Prediction Using Collaborative Filtering with Hierarchies and Side-Information. KDD 2011, pp. 141–149. doi:10.1145/2020408.2020436. – Dùng phân cấp để chia sẻ thông tin cho item thưa (làm trơn theo nút cha). <https://doi.org/10.1145/2020408.2020436>
- [DL01] GroupLens Research. MovieLens 100K Dataset (trang mô tả và tải dữ liệu). – Dữ liệu mặc định của bộ đề: tương tác người dùng–phim và nhãn thể loại. <https://grouplens.org/datasets/movielens/100k/>
- [DL02] GroupLens Research. README của MovieLens 100K (mô tả u.data, u.item, u.genre và điều khoản sử dụng). – Định dạng tệp, trường thể loại, điều khoản; số liệu kiểm tra: 100.000 đánh giá, 943 người dùng, 1.682 phim. <https://files.grouplens.org/datasets/movielens/ml-100k-README.txt>
- [CC01] PyTorch Documentation: Reproducibility (cố định seed, thuật toán tất định). – Cố định seed cho random, numpy, torch; ghi điều kiện tái lập. <https://docs.pytorch.org/docs/stable/notes/randomness.html>
- [CC14] pip Documentation: pip freeze. – Khóa phiên bản thư viện (requirements/lock) khi nộp. <https://pip.pypa.io/en/stable/cli/pip_freeze/>
- [BB07] Z. Meng, R. McCreadie, C. Macdonald, I. Ounis. Exploring Data Splitting Strategies for the Evaluation of Recommendation Models. RecSys 2020, pp. 681–686. doi:10.1145/3383313.3418479. – Ảnh hưởng của cách chia dữ liệu tới kết luận đánh giá. <https://doi.org/10.1145/3383313.3418479>
- [CC13] Python Documentation: hashlib. – Mã băm cho tệp split và giao thức (kiểm tra tái lập). <https://docs.python.org/3/library/hashlib.html>
- [BB04] K. Järvelin, J. Kekäläinen. Cumulated Gain-Based Evaluation of IR Techniques. ACM TOIS 20(4):422–446, 2002. doi:10.1145/582415.582418. – Định nghĩa DCG/NDCG dùng trong giao thức đánh giá. <https://doi.org/10.1145/582415.582418>
- [BB05] W. Krichene, S. Rendle. On Sampled Metrics for Item Recommendation. KDD 2020, pp. 1748–1757. doi:10.1145/3394486.3403226. – Lý do xếp hạng toàn bộ danh mục thay vì lấy mẫu 99 phim âm. <https://doi.org/10.1145/3394486.3403226>
- [CC12] pytest Documentation. – Viết kiểm thử tự động (ví dụ tay, bất biến, chống rò rỉ). <https://docs.pytest.org/en/stable/>
- [CC15] Google Colab. – Môi trường GPU miễn phí khi máy cá nhân không đủ. <https://colab.research.google.com/>
- [CC16] Kaggle Notebooks Documentation. – Môi trường GPU thay thế Colab. <https://www.kaggle.com/docs/notebooks>
- [BB01] S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009 (arXiv:1205.2618). – Hàm mất mát BPR, lấy mẫu âm, mô hình BPR-MF. <https://arxiv.org/abs/1205.2618>
- [CC05] pandas User Guide: Group by: split-apply-combine. – Đếm số tương tác, thống kê theo nhóm. <https://pandas.pydata.org/docs/user_guide/groupby.html>
- [CC11] Matplotlib Documentation. – Vẽ biểu đồ kết quả bằng mã (không vẽ tay). <https://matplotlib.org/stable/index.html>
