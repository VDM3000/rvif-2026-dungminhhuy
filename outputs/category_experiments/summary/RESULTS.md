# Kết quả thí nghiệm EMS theo category

> **Lưu ý:** Đây là phân tích exploratory đã tối ưu feature/model/alpha riêng cho
> từng category. Kết quả chính thức theo cấu hình cố định trong `message.txt` nằm
> tại `outputs/fixed_original_protocol/summary/RESULTS_FIXED_PROTOCOL.md`.

Đơn vị dưới đây là trung bình OOF của 4 fold chính thức. AUC là chỉ số chính.

| Category (số ảnh) | MSNet AUC | Behavior AUC | Fusion AUC | Fusion accuracy | Fusion F1 |
|---|---:|---:|---:|---:|---:|
| Manipulated (32) | 0.8444 | 0.8936 | 0.9235 | **0.8313** | 0.8181 |
| Natural (31) | 0.8217 | 0.8955 | 0.9181 | 0.7938 | 0.7925 |
| Social (22) | 0.8156 | **0.9061** | **0.9252** | 0.8250 | **0.8291** |
| Synthetic (15) | 0.7488 | 0.8719 | 0.8872 | 0.7813 | 0.7856 |

Theo AUC fusion, thứ hạng điểm là Social > Manipulated > Natural > Synthetic.
Social cũng đứng đầu behavior-only; Manipulated đứng đầu MSNet-only và fusion
accuracy. Kết quả toàn EMS trên Kaggle có fusion AUC 0.9503, nhưng chỉ nên dùng
làm mốc tham khảo vì nó dùng 100 ảnh và checkpoint MSNet pretrained chính thức.

Paired bootstrap 5.000 lượt, phân tầng theo fold và nhãn, cho chênh lệch AUC
Social trừ Manipulated là 0.0018 với CI 95% [-0.0384, 0.0436]. Không cặp category
nào có CI chênh lệch AUC fusion loại trừ 0. Vì vậy Social là category tốt nhất
theo điểm quan sát, nhưng chưa thể kết luận ưu thế có ý nghĩa thống kê.

Lưu ý chính: bốn category có 32/31/22/15 ảnh. Nên chạy thêm control cân bằng 15
ảnh/category với nhiều seed lấy mẫu hoặc đánh giá trên test set độc lập trước khi
đưa ra kết luận khoa học mạnh về bản chất category.
