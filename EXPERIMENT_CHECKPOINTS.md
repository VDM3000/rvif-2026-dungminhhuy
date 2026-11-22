# Checkpoint cho bốn đề xuất thí nghiệm

**Nguồn yêu cầu và quy tắc kỹ thuật:** [spec.md](spec.md). **Cập nhật:** 2026-11-22. File này là nơi theo dõi trạng thái; không suy ra đã chạy thí nghiệm từ việc có CSV lịch sử.

## Quy tắc tiếp tục phiên

1. Đọc `spec.md`, mục **Trạng thái và bàn giao** bên dưới, rồi kiểm tra Git status và file output thực tế.
2. Chọn một ID `OPEN`, ghi `IN_PROGRESS` và đường dẫn output riêng trước khi chạy. Mỗi run ghi `manifest.json` với input/hash, code commit, seed, fold, checkpoint, tham số và thời gian.
3. Chỉ đánh `DONE` khi đủ tiêu chí trong spec và điền bằng chứng ở bảng. Nếu dở, ghi bước hoàn tất cuối cùng, lỗi và lệnh tiếp tục. Không xóa hoặc ghi đè kết quả lịch sử.
4. Kết quả khác protocol phải nằm ở thư mục khác; thay đổi quyết định sau khi nhìn kết quả phải được ghi vào nhật ký và đánh dấu là phân tích mới.

## Checkpoint và phụ thuộc

| ID | Trạng thái | Việc và đầu ra cần có | Phụ thuộc |
|---|---|---|---|
| X-00 | DONE | Đã khôi phục `EMS_repo`, checkpoint, `.venv`; chạy suy luận MSNet thực và lưu hash đầu vào trong manifest baseline. | — |
| X-01 | DONE | Đã tái chạy seed 42, kiểm tra 160 người (4×40), đối chiếu ba bảng kết quả lịch sử và lưu frozen baseline + manifest. | X-00 |
| S-01 | DONE | Seed cố định 42/17/2026 cho HGB P+S; MSNet pretrained và bốn fold giữ nguyên. Mã và kiểm tra parity seed 42 ở `reviewer_experiments/seed_alpha.py`, `tests/test_seed_alpha.py`. | X-01 |
| S-02 | DONE | Ba run và 12 hàng seed×fold, prediction/alpha/manifest từng seed ở `outputs/reviewer_experiments/seed_alpha_final_2026-11-22/`. | S-01 |
| S-03 | DONE | `seed_metrics.csv`, `seed_summary.csv`: cả ba seed cho dự đoán y hệt, sample variance của mọi metric bằng 0; không phải CI. | S-02 |
| A-01 | DONE | `alpha_curve_by_fold.csv`: 84 điểm; 13 điểm theo fold có ΔAUC âm đáng kể (>1e-12), 0/21 điểm trung bình âm; α=0/0,8/1 khớp nguồn cũ. | X-01 |
| A-02 | DONE | `alpha_curve_mean.csv`, `alpha_delta_auc.png`, `interpretation.txt` ghi rõ validation, dấu âm và giới hạn external test. Đưa nhận xét vào `main.tex` tại R-01. | A-01 |
| L-01 | DONE | Đã khóa bốn lượt 80/40/40, MSNet 50 epoch cố định, batch 4, α và ngưỡng chỉ từ fold giữa. Cache 0,88 GiB khớp xác suất MSNet gốc trên 40 người Set_0; pilot một epoch chạy được trên GPU local. | X-01 |
| L-02 | DONE | Bốn MSNet huấn luyện từ đầu trên 80 train, 50 epoch cố định; chọn α/ngưỡng ở 40 người giữa; chấm 40 người cuối. Prediction/checkpoint/manifest tại `outputs/reviewer_experiments/late_211_2026-11-22/`. | L-01 |
| L-03 | DONE | `summary_211_2026-11-22/`: kiểm tra 4×40 ID duy nhất; MSNet mới AUC 77,80%, Late Fusion 84,92%, so với lịch sử có nhãn protocol khác. | L-02 |
| F-01 | DONE | Bốn file `msnet_embeddings.csv` mỗi file 160×56, ghép với 173 P+S thành 229 chiều; ID/shape và giá trị hữu hạn đã kiểm tra. | X-01 |
| F-02 | DONE | Head 229→32→1, ReLU/dropout 0,2, AdamW 0,001; chọn epoch bằng 40 người giữa; encoder MSNet giữ cố định, cùng split Late Fusion. | F-01, L-01 |
| F-03 | DONE | Early Fusion bốn lượt có head/preprocessing/checkpoint/prediction theo người; AUC 83,84% trên cùng 160 người, so với Late Fusion 84,92%. | F-02, L-03 |
| R-01 | DONE | Đã đối chiếu bảng/hình với output, cập nhật bài báo và tô xanh mọi phần thêm/sửa trong `main.tex`; PDF 7 trang đã biên dịch và kiểm tra màu. | S-03, A-02, L-03, F-03 |

## Quyết định phải ghi trước khi chạy

| ID | Quyết định cần chốt | Trạng thái |
|---|---|---|
| D-01 | MSNet huấn luyện từ đầu trên đúng 80 train, 50 epoch cố định, dùng checkpoint epoch cuối; không đọc fold chọn α hoặc fold đánh giá để chọn epoch. | Đã chốt; cần chạy kiểm chứng |
| D-02 | Early head 229→32→1, ReLU, dropout 0,2; AdamW 0,001, tối đa 100 epoch, chọn bằng AUC của 40 người giữa với patience 10. Encoder MSNet giữ cố định. | Đã chốt; cần chạy kiểm chứng |
| D-03 | Ngưỡng Otsu của Late/Early Fusion tính từ 40 người giữa sau khi khóa model; dùng nguyên ngưỡng đó cho 40 người đánh giá. | Đã chốt; kiểm tra `tests/test_protocol_211.py` |
| D-04 | Giữ đủ bốn lượt huấn luyện MSNet đúng 2/1/1. Cache 160 người (0,88 GiB) và pilot batch 4 chạy được trên GPU local; tiếp tục tại máy này, chỉ chuyển máy nếu run đầy đủ gặp giới hạn tài nguyên. | Đã chốt; pilot local đạt |

## Nhật ký bằng chứng

| Ngày giờ (Asia/Saigon) | ID | Input/commit/hash | Output và lệnh kiểm tra | Kết quả, lỗi hoặc giới hạn | Trạng thái mới |
|---|---|---|---|---|---|
| 2026-11-22 | SETUP | Notebook, `ems_overrides/MSNet.py`, `main.tex`, EMS README đã đọc. | `spec.md`, file này. | Chỉ lập đặc tả; chưa cài đặt/chạy/thay manuscript. | Tất cả OPEN |
| 2026-11-22 | X-00, X-01 | `outputs/reviewer_experiments/baseline_2026-11-22_seed42/manifest.json` ghi SHA-256 của notebook, file RINet, workbook, bốn checkpoint và `main.tex`, cùng commit repo/EMS. | `outputs/reviewer_experiments/baseline_2026-11-22_seed42/`: bản sao 96 file, gồm kết quả notebook và bốn fold `prob_val.txt`. Kiểm tra 160 ID duy nhất, mỗi fold 40; ba CSV chính khớp bản lịch sử sau khi chuẩn hóa CRLF/LF. | Kết quả tái tạo AUC BAMF 95,03%, MSNet 89,72%; đây là four-fold validation cũ, chưa phải xác nhận độc lập. | X-00, X-01 DONE |
| 2026-11-22 | S-01–S-03, A-01–A-02 | `reviewer_experiments/seed_alpha.py`; ba `manifest.json` chứa hash của dữ liệu, baseline manifest và mã; `tests/test_seed_alpha.py` 3/3 đạt. | `outputs/reviewer_experiments/seed_alpha_final_2026-11-22/`: 3×160 dự đoán, 12 hàng seed×fold, alpha curve 84 điểm và hình ΔAUC. | Seed 42 khớp từng xác suất lịch sử; seed 17 và 2026 giống hệt seed 42, sample variance 0. Mean ΔAUC không âm ở 21 α nhưng 13/84 điểm theo fold âm đáng kể; không có CI/external test. | S-01–S-03, A-01–A-02 DONE |
| 2026-11-22 | L-01, L-02 | `reviewer_experiments/prepare_cache.py`, `protocol_211.py`, `train_211.py`; `tests/test_protocol_211.py` 4/4 đạt. | `work/reviewer_experiments/cache/features_100.npy` shape 160×1056×100×14, 0,88 GiB; 40 người Set_0 cho xác suất khớp tuyệt đối với `prob_val.txt` gốc; pilot Set_0 một epoch kết thúc với exit 0. | Bốn lượt 50 epoch đang chạy; pilot một epoch không dùng làm số liệu bài báo. | L-01 DONE; L-02 IN_PROGRESS |
| 2026-11-22 | L-02, L-03, F-01–F-03 | `reviewer_experiments/train_211.py`, `early_211.py`, `summarize_211.py`; các bài kiểm tra protocol/early/summary đạt. | `outputs/reviewer_experiments/late_211_2026-11-22/`, `early_211_2026-11-22/`, `summary_211_2026-11-22/`; mỗi phương pháp có 4×40 đánh giá, 160 ID duy nhất. | Mean AUC: MSNet mới 77,80%, Late 84,92%, Early 83,84%. Late α lần lượt 0/0,10/0,05/0,35; Late sửa đúng thêm 26 người và sai thêm 18 so với MSNet mới. Khác protocol lịch sử, không phải hidden test. | L-02–L-03, F-01–F-03 DONE; R-01 IN_PROGRESS |
| 2026-11-22 | R-01 | Đối chiếu `main.tex` với bản gốc SHA-256 trong baseline; số liệu lấy từ `summary_211_2026-11-22/` và `seed_alpha_final_2026-11-22/`. | `../2026_RVIF_HuyMinhDung_Final/main.tex`; snapshot cho nhóm là `paper/main.tex` và `paper/main.pdf` (PDF local gốc ở `work/reviewer_experiments/manuscript_preview/main.pdf`). Chạy 12 bài kiểm tra mã: đạt; biên dịch Tectonic 0.17.0: exit 0. Diff 25 dòng nội dung thêm/sửa, không có dòng mới ngoài `\textcolor{blue}` hoặc bảng `\color{blue}`. PDF 7 trang có các đoạn màu xanh ở trang 1, 4–7. | Bảng mới đã thu gọn, không còn cảnh báo dòng quá rộng ở bảng này. Hai bảng cũ vẫn có cảnh báo dòng quá rộng 34,53 pt và 5,39 pt; chưa sửa vì không thuộc phần bổ sung. PDF là bản kiểm tra, chưa phải bản nộp. | R-01 DONE |

## Trạng thái và bàn giao hiện tại

- **Đã hoàn thành:** toàn bộ checkpoint X/S/A/L/F/R trong spec. `main.tex` đã được tô xanh phần thêm/sửa và biên dịch thành PDF kiểm tra 7 trang.
- **Bước tiếp theo:** nếu chuẩn bị nộp bài, rà soát nội dung khoa học và giới hạn trang với đồng tác giả; nếu sửa bài tiếp, so với bản gốc trong baseline để tiếp tục tô xanh đúng phần thay đổi. Kiểm tra Git status, output và manifest trước khi tiếp tục.
- **Hiện có:** `EMS/`, `EMS_repo/`, `.venv/`, `model_parameters/`, `feature_dict_RINet.npy`, dữ liệu theo người từ lần chạy seed 42 và frozen baseline ở `outputs/reviewer_experiments/baseline_2026-11-22_seed42/`.
- **Hiện có thêm:** ba seed và phân tích α ở `outputs/reviewer_experiments/seed_alpha_final_2026-11-22/`. Hai thư mục thử nghiệm `seed_alpha_2026-11-22/` và `seed_alpha_2026-11-22_verified/` là bản trung gian, không dùng để trích số; chỉ dùng thư mục `seed_alpha_final_2026-11-22/`.
- **Hiện có thêm:** bản PDF kiểm tra tại `work/reviewer_experiments/manuscript_preview/main.pdf`, snapshot công bố tại `paper/main.pdf`. Cache 0,88 GiB và bốn checkpoint MSNet mới đã chạy trên máy hiện tại; không dùng checkpoint pretrained ba-fold cũ cho 2/1/1. Bài chưa được đồng tác giả duyệt để nộp.
- **Quy tắc sửa bài:** giữ `main.tex` hiện tại làm mốc (SHA-256 trong manifest baseline); khi thực hiện R-01, mọi đoạn viết thêm hoặc sửa phải được tô màu xanh trong LaTeX bằng `\textcolor{blue}{...}` hoặc macro tương đương. Kiểm tra bản PDF để chắc chắn màu hiện ra đúng và không tô xanh nội dung không đổi.
- **Không được nhầm:** α fusion với `alpha` nội bộ MSNet; ba seed với CI; fold đánh giá 2/1/1 với hidden/external test; head gốc 56 chiều với MLP 229 chiều mới.
- **Worktree có trước khi lập spec:** các tài liệu đang được chuyển sang `docs/`; `EMS_repo` có trạng thái Git `D`. Không dùng lệnh reset để xóa thay đổi này.
