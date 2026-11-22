# BAMF trên bộ dữ liệu EMS

Repository này chứa mã và kết quả cho ba luồng nghiên cứu. Bản thảo bài báo nằm ở thư mục **cùng cấp** `../2026_RVIF_HuyMinhDung_Final/`; không nằm trong repository mã.

1. **Kết quả BAMF chính:** `notebooks/windows.ipynb` tạo lại dự đoán MSNet pretrained trên bốn fold, trích đặc trưng hành vi, đánh giá 21 cấu hình và ghép hai điểm dự đoán. `notebooks/original_kaggle.ipynb` là bản cho Kaggle. Bảng tổng hợp lịch sử nằm ở `improve_behavioral_final_results/`.
2. **So sánh bốn nhóm ảnh:** mã trong `category_experiments/`. Kết quả với cấu hình cố định ở `outputs/fixed_original_protocol/`; kết quả khám phá tối ưu riêng từng nhóm ở `outputs/category_experiments/summary/`. Xem [giao thức cố định](docs/experiments/FIXED_PROTOCOL.md) và [hướng dẫn nhóm ảnh](docs/experiments/CATEGORY_EXPERIMENTS.md).
3. **Thử 20/40/60 ảnh cân bằng:** mã trong `balanced_stimulus_experiment/`, kết quả trong `outputs/balanced_stimulus_experiment/`. Xem [hướng dẫn số lượng ảnh](docs/experiments/BALANCED_STIMULUS_EXPERIMENT.md).

Danh mục chức năng từng file và thư mục: [FILE_MAP.md](FILE_MAP.md). [Đặc tả bốn thí nghiệm theo góp ý GVHD](spec.md) và [checkpoint triển khai](EXPERIMENT_CHECKPOINTS.md) theo dõi công việc mới. Yêu cầu nguyên bản và kế hoạch cũ được lưu trong `docs/history/` để đối chiếu, không phải lệnh chạy hiện tại.

## Dữ liệu và mã EMS gốc

- `EMS/` là dữ liệu local. `EMS/README.txt` mô tả 100 ảnh, 160 người train/valid và 48 người test không có nhãn khả dụng trong dự án này.
- `EMS_repo/` là submodule của [repo EMS gốc](https://github.com/YingjieSong1/EMS). Nó hiện chưa có trong worktree này; Git ghi trạng thái `D EMS_repo`. Khôi phục bằng `git submodule update --init --recursive` trước khi chạy notebook.
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

Notebook chính không huấn luyện lại MSNet bằng `train.py`. Kết quả AUC 95,03% được báo trong bài là kết quả four-fold validation lịch sử; việc tái chạy để đối chiếu kết quả khác với một đánh giá độc lập mới. Notebook hiện chưa được chạy lại thành công trên trạng thái thư mục hiện tại.

## Thí nghiệm bổ sung

Sau khi chuẩn bị môi trường, có thể dùng các lệnh riêng:

```powershell
.\run_fixed_protocol.ps1
.\run_category_experiments.ps1
.\run_balanced_stimulus_experiment.ps1
```

Chúng chạy các nghiên cứu nhóm/số lượng ảnh, không thay thế cho việc chạy `notebooks/windows.ipynb`. Kết quả `outputs/category_experiments/summary/` đã tối ưu riêng theo nhóm ảnh, nên phải phân biệt với `outputs/fixed_original_protocol/` khi viết bài.

## Lưu dữ liệu và kết quả

`.gitignore` loại dữ liệu EMS, `.npy`, checkpoint, môi trường Python và nhiều output theo người khỏi Git. Các bảng tổng hợp đã có trong `outputs/` và `improve_behavioral_final_results/` là kết quả lịch sử. Trước khi chạy lại notebook, sao lưu `work/result_pretrained_backup/` và `outputs/improve_behavioral/` nếu chúng tồn tại, vì một số cell ghi lại hoặc thay thế file trong các thư mục đó.
