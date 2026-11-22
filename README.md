# BAMF trên bộ dữ liệu EMS

Repository này chứa mã và kết quả cho ba luồng nghiên cứu. `paper/` là bản sao của bản thảo ở thư mục làm việc cùng cấp `../2026_RVIF_HuyMinhDung_Final/` tại lần công bố kết quả ngày 2026-11-22; `paper/main.pdf` là PDF kiểm tra đã biên dịch. Khi sửa bản thảo tiếp, đồng bộ cả hai nơi trước khi push.

1. **Kết quả BAMF chính:** `notebooks/windows.ipynb` tạo lại dự đoán MSNet pretrained trên bốn fold, trích đặc trưng hành vi, đánh giá 21 cấu hình và ghép hai điểm dự đoán. `notebooks/original_kaggle.ipynb` là bản cho Kaggle. Bảng tổng hợp lịch sử nằm ở `improve_behavioral_final_results/`.
2. **So sánh bốn nhóm ảnh:** mã trong `category_experiments/`. Kết quả với cấu hình cố định ở `outputs/fixed_original_protocol/`; kết quả khám phá tối ưu riêng từng nhóm ở `outputs/category_experiments/summary/`. Xem [giao thức cố định](docs/experiments/FIXED_PROTOCOL.md) và [hướng dẫn nhóm ảnh](docs/experiments/CATEGORY_EXPERIMENTS.md).
3. **Thử 20/40/60 ảnh cân bằng:** mã trong `balanced_stimulus_experiment/`, kết quả trong `outputs/balanced_stimulus_experiment/`. Xem [hướng dẫn số lượng ảnh](docs/experiments/BALANCED_STIMULUS_EXPERIMENT.md).

Danh mục chức năng từng file và thư mục: [FILE_MAP.md](FILE_MAP.md). [Đặc tả bốn thí nghiệm theo góp ý GVHD](spec.md) và [checkpoint triển khai](EXPERIMENT_CHECKPOINTS.md) theo dõi công việc mới. Yêu cầu nguyên bản và kế hoạch cũ được lưu trong `docs/history/` để đối chiếu, không phải lệnh chạy hiện tại.

## Dữ liệu và mã EMS gốc

- `EMS/` là dữ liệu local. `EMS/README.txt` mô tả 100 ảnh, 160 người train/valid và 48 người test không có nhãn khả dụng trong dự án này.
- `EMS_repo/` là submodule của [repo EMS gốc](https://github.com/YingjieSong1/EMS). Trên máy hiện tại submodule đã được khôi phục và bốn file gốc có thay đổi local do `apply_ems_overrides.ps1`; người clone repo mới cần chạy `git submodule update --init --recursive`, rồi áp dụng overrides.
- `ems_overrides/` có bốn bản điều chỉnh cho MSNet, train, validation và test. `apply_ems_overrides.ps1` chép chúng vào `EMS_repo/`; thao tác này làm thay đổi nội dung submodule trong worktree local.
- `feature_dict_RINet.npy` là bản đồ đặc trưng ảnh khoảng 10,88 GiB, không phải checkpoint. `model_parameters.zip` chứa bốn checkpoint `Set_0` đến `Set_3`; giải nén trực tiếp ra `model_parameters/` trước khi chạy. Hai nguồn này không được Git theo dõi.

## Tái chạy kết quả BAMF chính trên Windows

Đọc [README_WINDOWS.md](README_WINDOWS.md) cho các bước chuẩn bị, lệnh kiểm tra và lưu ý RAM. Tóm tắt luồng:

```text
EMS + RINet features + MSNet pretrained checkpoints
  → EMS_repo/val.py tạo prob_val.txt cho bốn fold
  → notebook trích đặc trưng hành vi, thử 21 cấu hình
  → notebook ghép điểm và ghi bảng kết quả
```

Notebook chính không huấn luyện lại MSNet bằng `train.py`. Kết quả AUC 95,03% được báo trong bài là kết quả four-fold validation lịch sử; việc tái chạy để đối chiếu kết quả khác với một đánh giá độc lập mới. Notebook đã chạy lại thành công trên máy hiện tại; bản sao kết quả và hash đầu vào nằm trong `outputs/reviewer_experiments/baseline_2026-11-22_seed42/`.

## Thí nghiệm bổ sung

Sau khi chuẩn bị môi trường, có thể dùng các lệnh riêng:

```powershell
.\run_fixed_protocol.ps1
.\run_category_experiments.ps1
.\run_balanced_stimulus_experiment.ps1
```

Chúng chạy các nghiên cứu nhóm/số lượng ảnh, không thay thế cho việc chạy `notebooks/windows.ipynb`. Kết quả `outputs/category_experiments/summary/` đã tối ưu riêng theo nhóm ảnh, nên phải phân biệt với `outputs/fixed_original_protocol/` khi viết bài.

## Lưu dữ liệu và kết quả

`.gitignore` vẫn loại dữ liệu EMS gốc, file đặc trưng RINet 11 GB, môi trường Python và cache `work/`. Bản công bố cho nhóm gồm toàn bộ `outputs/improve_behavioral/` và `outputs/reviewer_experiments/`, kể cả dự đoán theo người và checkpoint nhỏ của thí nghiệm 2/1/1. Các output mới trong các thư mục này sẽ hiện trong Git status. Trước khi chạy lại notebook, sao lưu `work/result_pretrained_backup/` và `outputs/improve_behavioral/` nếu chúng tồn tại, vì một số cell ghi lại hoặc thay thế file trong các thư mục đó.
