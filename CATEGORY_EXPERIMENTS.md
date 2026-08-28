# Thí nghiệm EMS theo 4 nhóm ảnh

Pipeline này giữ 4 fold trong `EMS/Train_Valid.xlsx`, seed 42, batch size 8,
50 epoch, đặc trưng RINet và cách đánh giá Otsu/macro 4-fold của notebook gốc.
MSNet được huấn luyện lại từ đầu cho từng category vì checkpoint gốc có tầng
`Linear(100, 1)` và không thể đánh giá hợp lệ trên một tập con ảnh.

Số ảnh sử dụng: Manipulated 32, Natural 31, Social 22, Synthetic 15. Vì số ảnh
không bằng nhau, kết quả trả lời câu hỏi “toàn bộ ảnh hiện có của category nào
tốt nhất”; nó chưa tách riêng ảnh hưởng của loại ảnh khỏi ảnh hưởng của số ảnh.

## Chạy lại toàn bộ

```powershell
.\run_category_experiments.ps1
```

Hoặc chạy từng bước:

```powershell
.\.venv\Scripts\python.exe .\category_experiments\prepare_cache.py --category all
.\.venv\Scripts\python.exe .\category_experiments\train_msnet.py --category all --fold all --epochs 50 --batch-size 8
.\.venv\Scripts\python.exe .\category_experiments\run_behavior_fusion.py --category all
.\.venv\Scripts\python.exe .\category_experiments\analyze_results.py --bootstrap-iterations 5000
```

Các fold đã có `completed.json` sẽ tự động được bỏ qua. Cache nằm trong
`work/category_experiments/cache`; checkpoint, xác suất OOF, metric và bảng tổng
hợp nằm trong `outputs/category_experiments`.

## Cách đọc kết quả

- `summary/category_comparison_metrics.csv`: cả ba phương pháp cho bốn category.
- `summary/best_category_by_method.csv`: category có AUC cao nhất theo phương pháp.
- `summary/fusion_auc_pairwise_bootstrap.csv`: CI 95% của chênh lệch AUC fusion.
- `summary/category_plus_full_ems_reference.csv`: có thêm kết quả Kaggle toàn EMS;
  đây chỉ là mốc tham khảo vì MSNet toàn EMS dùng checkpoint pretrained chính thức.
- `summary/category_auc_comparison.png`: biểu đồ AUC.

Bootstrap được phân tầng theo fold và nhãn, ghép cặp cùng subject giữa category.
CI sau khi đã chọn mô hình/category tốt nhất có thể lạc quan; nên xem đây là phân
tích thăm dò. Kiểm chứng mạnh hơn tiếp theo là chạy control cân bằng 15 ảnh cho
mỗi category với nhiều lần lấy mẫu và/hoặc dùng một test set độc lập.
