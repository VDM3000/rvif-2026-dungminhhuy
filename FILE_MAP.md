# Bản đồ thư mục và file của dự án

Phạm vi: `dav/rvif-2026-dungminhhuy/`. Bản đồ này được đối chiếu với file thực có ngày 2026-11-22. Thư mục bản thảo `dav/2026_RVIF_HuyMinhDung_Final/` nằm ngoài phạm vi sắp xếp. Các file dữ liệu cùng mẫu tên được giải thích theo nhóm; xem `EMS/README.txt` để biết định nghĩa từng bản ghi. `bamf.zip` không tồn tại và không thuộc cấu trúc hiện tại.

## Root `dav/`

| Mục | Ý nghĩa |
|---|---|
| `2026_RVIF_HuyMinhDung_Final/` | Bản thảo và tài liệu đi kèm; không thay đổi. |
| `rvif-2026-dungminhhuy/` | Repository mã nguồn, dữ liệu local và kết quả thí nghiệm. `.git` nằm tại đây. |

## Root repository

| File/thư mục | Ý nghĩa |
|---|---|
| `.git/` | Lịch sử Git và trạng thái phiên bản. |
| `.gitignore` | Loại dữ liệu EMS, file model lớn, môi trường và output nhạy cảm khỏi Git. |
| `.gitattributes` | Quy tắc xử lý file của Git. |
| `.gitmodules` | Ghim repo EMS gốc làm submodule `EMS_repo/`. |
| `README.md` | Giới thiệu thí nghiệm và lệnh chạy. |
| `README_WINDOWS.md` | Chuẩn bị máy Windows và chạy notebook chính. |
| `FILE_MAP.md` | Bản đồ này. |
| `spec.md` | Đặc tả bốn thí nghiệm mới theo góp ý GVHD và giới hạn phương pháp. |
| `EXPERIMENT_CHECKPOINTS.md` | Tiến độ, quyết định và bàn giao cho bốn thí nghiệm trong `spec.md`. |
| `requirements-windows.txt` | Gói Python bổ sung cho môi trường Windows. |
| `setup_windows.ps1` | Tạo `.venv`, cài gói, đăng ký kernel và kiểm tra đầu vào. |
| `apply_ems_overrides.ps1` | Chép bốn bản điều chỉnh từ `ems_overrides/` vào `EMS_repo/`. |
| `run_category_experiments.ps1` | Lệnh chạy thí nghiệm theo bốn nhóm ảnh. |
| `run_fixed_protocol.ps1` | Lệnh chạy và kiểm tra thí nghiệm nhóm ảnh với cấu hình cố định. |
| `run_balanced_stimulus_experiment.ps1` | Lệnh chạy thí nghiệm 20/40/60 ảnh cân bằng. |
| `feature_dict_RINet.npy` | Bản đồ đặc trưng ảnh do RINet tạo sẵn, khoảng 10,88 GiB; đầu vào MSNet, không phải trọng số. |
| `model_parameters.zip` | Bản nén checkpoint MSNet pretrained; khi dùng cần giải nén ra `model_parameters/`. |
| `EMS_repo/` | Submodule EMS gốc, hiện vắng mặt trong worktree; notebook cần thư mục này. Đây là đường dẫn dự kiến, không phải thư mục hiện có. |
| `.venv/` | Môi trường Python local sẽ được tạo bởi setup; hiện chưa có. |
| `work/` | Kết quả MSNet tạm và bản sao `prob_val.txt` sẽ được notebook tạo; hiện chưa có. |

## Dữ liệu `EMS/`

| Mục | Ý nghĩa |
|---|---|
| `README.txt` | Mô tả bộ dữ liệu và cấu trúc. |
| `UsageAgreement.txt` | Điều kiện sử dụng dữ liệu. |
| `Train_Valid.xlsx` | Danh sách 160 người và phân bổ bốn fold. |
| `Train_Valid/Fixations/*.xlsx` | 160 file bản ghi mắt nhìn có nhãn dùng cho huấn luyện và đánh giá. |
| `Test/Fixations/*.xlsx` | 48 file mắt nhìn test không có nhãn khả dụng trong dự án này. |
| `Images/{Manipulated Images,Natural Scenes,Social Scenes,Synthetic Images}/*.jpg` | 100 ảnh kích thích chia bốn nhóm. |

## Notebook và mã EMS

| File/thư mục | Ý nghĩa |
|---|---|
| `notebooks/original_kaggle.ipynb` | Bản notebook gốc chạy trên Kaggle, giữ để đối chiếu logic. |
| `notebooks/windows.ipynb` | Notebook chính để tạo lại MSNet probabilities, 21 cấu hình behavioral và BAMF fusion trên Windows. |
| `ems_overrides/MSNet.py` | Bản MSNet điều chỉnh cho số lượng ảnh biến đổi. |
| `ems_overrides/train.py` | Bản lệnh huấn luyện MSNet điều chỉnh. |
| `ems_overrides/val.py` | Bản lệnh đánh giá fold, nạp `.npy` và ghi `prob_val.txt`. |
| `ems_overrides/test.py` | Bản lệnh tạo `prob_test.txt` trên 48 người không nhãn. |
| `tools/verify_local_setup.py` | Kiểm tra dữ liệu, checkpoint, gói Python và một lượt forward nhỏ. |
| `tools/smoke_real_inference.py` | Kiểm tra suy luận MSNet thực với file RINet lớn. |

## Mã thí nghiệm theo nhóm ảnh

| File | Ý nghĩa |
|---|---|
| `category_experiments/__init__.py` | Đánh dấu Python package. |
| `category_experiments/common.py` | Cấu hình và hàm chung. |
| `category_experiments/prepare_cache.py` | Chuẩn bị dữ liệu/cache theo nhóm ảnh. |
| `category_experiments/train_msnet.py` | Huấn luyện MSNet theo nhóm ảnh. |
| `category_experiments/run_behavior_fusion.py` | Chạy nhánh hành vi và ghép điểm theo nhóm. |
| `category_experiments/run_fixed_protocol.py` | Chạy bốn nhóm với cấu hình BAMF cố định. |
| `category_experiments/analyze_results.py` | Tổng hợp phân tích kết quả exploratory. |
| `category_experiments/validate_fixed_protocol.py` | Kiểm tra đầu ra protocol cố định. |

## Mã thí nghiệm số lượng ảnh

| File | Ý nghĩa |
|---|---|
| `balanced_stimulus_experiment/__init__.py` | Đánh dấu Python package. |
| `balanced_stimulus_experiment/common.py` | Cấu hình và hàm chung. |
| `balanced_stimulus_experiment/prepare.py` | Chọn và chuẩn bị tập 20/40/60 ảnh. |
| `balanced_stimulus_experiment/train_msnet.py` | Huấn luyện MSNet cho từng tập ảnh. |
| `balanced_stimulus_experiment/run_fixed_protocol.py` | Chạy behavioral fusion với cấu hình cố định. |
| `balanced_stimulus_experiment/validate.py` | Kiểm tra kết quả thí nghiệm. |

## Kết quả đã lưu

| Mục | Ý nghĩa |
|---|---|
| `improve_behavioral_final_results/final_comparison_metrics.csv` | So sánh MSNet, behavioral và BAMF cho kết quả chính lịch sử. |
| `improve_behavioral_final_results/best_fusion_foldwise_metrics.csv` | Kết quả fusion từng fold. |
| `improve_behavioral_final_results/best_fusion_alpha_by_fold.csv` | Alpha được chọn theo fold. |
| `improve_behavioral_final_results/feature_sets_clean.json` | Danh sách cột thuộc từng nhóm đặc trưng hành vi. |
| `outputs/category_experiments/summary/` | Kết quả exploratory tối ưu riêng từng category: bảng so sánh, best category, CI/bootstrap theo category, hình và `RESULTS.md`. Không thay cho thí nghiệm cấu hình cố định. |
| `outputs/fixed_original_protocol/` | Kết quả bốn category khi giữ cố định cấu hình: `protocol.json`, ánh xạ ảnh, báo cáo kiểm tra, từng category và `summary/`. |
| `outputs/fixed_original_protocol/{manipulated,natural,social,synthetic}/` | Mỗi thư mục có `feature_columns.json`, `category_summary.csv` và ba file `behavior_foldwise_metrics.csv`, `msnet_foldwise_metrics.csv`, `fusion_foldwise_metrics.csv`. |
| `outputs/fixed_original_protocol/summary/` | `auc_by_fold.csv`, `best_category_by_criterion.csv`, `final_results_with_baseline.csv`, hình so sánh và báo cáo `RESULTS_FIXED_PROTOCOL.md`. |
| `outputs/balanced_stimulus_experiment/manifests/stimulus_{20,40,60}.csv` | Danh sách ảnh được chọn ở từng mức. |
| `outputs/balanced_stimulus_experiment/results_{20,40,60}/` | `feature_columns.json` và `fold_metrics.csv` cho mỗi mức ảnh. |
| `outputs/balanced_stimulus_experiment/summary/` | Bảng tổng hợp, kết quả fold, AUC, hình và `experiment_summary.md`. |
| `outputs/balanced_stimulus_experiment/{protocol.json,validation_report.json}` | Cấu hình thí nghiệm và báo cáo kiểm tra. |

## Tài liệu được sắp xếp lại

| File | Ý nghĩa |
|---|---|
| `docs/experiments/FIXED_PROTOCOL.md` | Giải thích thí nghiệm category với cấu hình cố định. |
| `docs/experiments/CATEGORY_EXPERIMENTS.md` | Hướng dẫn thí nghiệm theo nhóm ảnh. |
| `docs/experiments/BALANCED_STIMULUS_EXPERIMENT.md` | Hướng dẫn thí nghiệm 20/40/60 ảnh. |
| `docs/history/message.txt` | Yêu cầu gốc cho thí nghiệm category. |
| `docs/history/message2.txt` | Yêu cầu gốc cho thí nghiệm số lượng ảnh. |
| `docs/history/plan_2.md` | Kế hoạch chi tiết cũ, giữ làm lịch sử. |
| `docs/history/prepare_local_notebook.py` | Script từng sinh notebook Windows từ tên file Kaggle cũ; đã lỗi thời, không còn được setup gọi. |

## Trạng thái cần biết trước khi chạy lại

Kiểm kê ngày 2026-11-22: `EMS/` có 311 file, `outputs/` có 51 file, `notebooks/` có 2 file, `docs/` có 7 file. `EMS_repo/`, `.venv/`, `work/` và `model_parameters/` chưa có trong worktree hiện tại. `setup_windows.ps1` đã trỏ tới notebook có thật, nhưng chưa có lần chạy mới được xác nhận. Các kết quả trong `outputs/` và `improve_behavioral_final_results/` là artifact lịch sử, không phải bằng chứng rằng môi trường hiện nay đã chạy được từ đầu.
