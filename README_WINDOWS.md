# Chạy thí nghiệm EMS trên Windows

Notebook gốc `notebookaaf7c6b835.ipynb` được giữ nguyên. Bản dùng cho máy này là
`notebook_windows.ipynb`.

## Cài đặt một lần

Mở PowerShell tại thư mục dự án và chạy:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

Script tạo `.venv`, cài PyTorch CUDA cho RTX 4050, cài các thư viện, tạo kernel
`RVIF EMS (Windows GPU)`, sinh notebook Windows và chạy smoke test.

## Mở notebook

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

Mở `notebook_windows.ipynb`, chọn kernel `RVIF EMS (Windows GPU)`, rồi chạy từ
trên xuống dưới. Kết quả được ghi vào `outputs/improve_behavioral`; file tạm và
kết quả MSNet nằm trong `work`.

## Lưu ý bộ nhớ

`feature_dict_RINet.npy` khoảng 11.1 GB. Notebook đã đặt `num_workers=0` để
Windows không nhân bản dữ liệu cho các worker. Trước khi chạy phần MSNet, nên
đóng ứng dụng dùng nhiều RAM. Không chạy song song nhiều kernel notebook.

Kiểm tra lại môi trường bất kỳ lúc nào:

```powershell
.\.venv\Scripts\python.exe .\tools\verify_local_setup.py
```

Kiểm tra một inference thật (lệnh này nạp feature dictionary 10.88 GB):

```powershell
.\.venv\Scripts\python.exe .\tools\smoke_real_inference.py
```
