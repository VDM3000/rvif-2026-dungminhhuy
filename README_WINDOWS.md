# Chạy thí nghiệm BAMF trên Windows

Notebook chuẩn để tái chạy kết quả chính là `notebooks/windows.ipynb`. Bản `notebooks/original_kaggle.ipynb` là nguồn tham chiếu cho môi trường Kaggle.

## Đầu vào cần có

- `EMS/`: dữ liệu EMS với `Train_Valid.xlsx`, 160 bản ghi train/valid, 48 bản ghi test và 100 ảnh.
- `EMS_repo/`: submodule mã nguồn EMS ở commit đã ghim trong `.gitmodules`; áp dụng `apply_ems_overrides.ps1` sau khi khôi phục.
- `feature_dict_RINet.npy`: bản đồ đặc trưng ảnh đã tạo bởi RINet, khoảng 10,88 GiB.
- `model_parameters/Set_0` đến `Set_3/best_model.pt`: trọng số MSNet pretrained. ZIP hiện có chứa trực tiếp bốn thư mục `Set_*`.

## Chuẩn bị và chạy

Tại thư mục repository trong PowerShell:

```powershell
git submodule update --init --recursive
.\apply_ems_overrides.ps1
Expand-Archive -LiteralPath .\model_parameters.zip -DestinationPath .\model_parameters
.\setup_windows.ps1
.\.venv\Scripts\python.exe -m jupyter lab
```

Chỉ giải nén khi `model_parameters/` chưa tồn tại; nếu đã có, kiểm tra đủ bốn `best_model.pt` trước khi quyết định có thay thế file nào. `setup_windows.ps1` cài thư viện nên cần dung lượng đĩa và mạng. Hiện `EMS_repo/`, `model_parameters/` và `.venv/` chưa có; các lệnh trên chưa được xác minh bằng một lần chạy hoàn chỉnh trên máy này.

Mở `notebooks/windows.ipynb`, chọn kernel `RVIF EMS (Windows GPU)` và chạy cell từ trên xuống. `setup_windows.ps1` tạo môi trường và kiểm tra đầu vào; nó không chạy toàn bộ notebook. Notebook dùng checkpoint MSNet có sẵn để tạo `prob_val.txt`, sau đó huấn luyện nhánh behavioral và tính fusion. Thí nghiệm chính không gọi `train.py` để huấn luyện lại MSNet.

Kết quả notebook ghi dưới `outputs/improve_behavioral/`; file MSNet tạm nằm dưới `work/`. Các file này bị Git bỏ qua. Trước khi tái chạy, lưu riêng bản kết quả lịch sử và lưu ý cell 8 thay thế `work/result_pretrained_backup` nếu thư mục này tồn tại.

Kiểm tra môi trường:

```powershell
.\.venv\Scripts\python.exe .\tools\verify_local_setup.py
```

Kiểm tra một lượt suy luận thực (nạp file đặc trưng lớn):

```powershell
.\.venv\Scripts\python.exe .\tools\smoke_real_inference.py
```

File `.npy` lớn có thể cần nhiều RAM hơn dung lượng file; đóng ứng dụng nặng và chỉ chạy một kernel notebook. Nếu đã lưu đủ `prob_val.txt` của bốn fold và chỉ cần tính lại behavioral/fusion, có thể bắt đầu từ bảng xác suất đó sau khi xác minh đúng người và fold, nhưng notebook hiện tại được thiết kế để tái tạo chúng từ đầu.
