# Giao thức cố định từ message.txt

Đây là pipeline chính thức cho câu hỏi nghiên cứu theo category. Cả bốn category
dùng đúng một cấu hình:

```text
MSNet + Pupil/Scanpath + HistGradientBoosting
p_fusion = 0.8 * p_msnet + 0.2 * p_behavior
```

Chạy lại và kiểm định bằng:

```powershell
.\run_fixed_protocol.ps1
```

Pipeline fit lại HGB trên ba fold và dự đoán fold còn lại, lặp đủ bốn fold. Nó
dùng dự đoán OOF từ 16 MSNet category đã huấn luyện. Threshold Otsu được tính
độc lập trong từng validation fold, giống pipeline gốc.

Kết quả chính nằm tại `outputs/fixed_original_protocol/summary`. Kết quả cũ tại
`outputs/category_experiments/summary` là phân tích exploratory có tối ưu riêng
feature/model/alpha theo category và không phải bảng chính thức của giao thức này.
