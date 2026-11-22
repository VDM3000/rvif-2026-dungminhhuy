# Kế Hoạch Thí Nghiệm Cân Bằng Số Lượng Ảnh EMS

> **Dành cho người triển khai bằng agent:** REQUIRED SUB-SKILL: dùng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để triển khai kế hoạch này theo từng phần. Các bước dùng checkbox (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Mô tả chi tiết thí nghiệm mới được pull về, dùng các tập 20, 40 và 60 ảnh cân bằng giữa 4 nhóm ảnh để xem giảm số ảnh đầu vào làm kết quả phân biệt SZ/HC thay đổi thế nào so với baseline 100 ảnh.

**Kiến trúc:** Thí nghiệm mới nằm trong `balanced_stimulus_experiment/`. Nó không hỏi "category nào tốt nhất" như `plan_1.md`; nó hỏi "cần bao nhiêu ảnh, nếu mỗi category đóng góp số ảnh bằng nhau, để kết quả vẫn gần baseline 100 ảnh". Code tạo 3 tập ảnh lồng nhau 20/40/60, huấn luyện MSNet theo từng số lượng ảnh, chạy lại phần hành vi mắt nhìn bằng đúng cấu hình baseline, rồi ghép điểm theo công thức cố định `0.8 * MSNet + 0.2 * behavior`.

**Công nghệ:** Python, PowerShell, PyTorch, NumPy, Pandas, scikit-learn, OpenCV, Matplotlib, EMS dataset, RINet feature cache, MSNet.

**Yêu cầu:** Tạo `plan_2.md` ở root dự án, giải thích thật chi tiết các thay đổi mới pull về để người chưa biết dự án vẫn hiểu ý nghĩa và cách làm.

## Ràng Buộc Chung

- Baseline 100 ảnh lấy từ `improve_behavioral_final_results`, không chạy lại hoặc chỉnh lại baseline trong thí nghiệm này.
- Thí nghiệm mới chỉ dùng validation 160 người trong `Train_Valid.xlsx`, không được ghi thành kết quả test ẩn 48 người.
- Giữ cố định công thức ghép điểm: `p_fusion = 0.8 * p_msnet + 0.2 * p_behavior`.
- Giữ cố định phần hành vi mắt nhìn: `Pupil + Scanpath`.
- Giữ cố định bộ học hành vi: `HistGradientBoosting`.
- Giữ cố định 4 fold: `Set_0`, `Set_1`, `Set_2`, `Set_3`.
- Giữ seed chọn ảnh là `42`.
- Kết quả hiện tại là một lần chọn ảnh duy nhất theo seed 42; chưa đo độ dao động nếu chọn ảnh khác.
- Không commit dữ liệu nặng như cache `.npy`, checkpoint `.pt`, dataset hoặc file train trung gian lớn nếu không có yêu cầu riêng.

---

## 0. Thí Nghiệm Mới Là Gì?

### 0.1. Khác gì với thí nghiệm category trước đó?

Thí nghiệm category trước đó hỏi:

```text
Nếu chỉ dùng từng nhóm ảnh riêng lẻ,
Natural / Social / Synthetic / Manipulated nhóm nào phân biệt SZ/HC tốt nhất?
```

Thí nghiệm mới hỏi câu khác:

```text
Nếu giảm số lượng ảnh từ 100 xuống 60, 40, 20,
nhưng vẫn lấy đều ảnh từ cả 4 category,
kết quả phân biệt SZ/HC giảm bao nhiêu?
```

Nói ngắn:

```text
plan_1.md: so sánh loại ảnh.
plan_2.md: so sánh số lượng ảnh, nhưng giữ cân bằng loại ảnh.
```

### 0.2. Vì sao cần thí nghiệm cân bằng?

Trong EMS, số ảnh mỗi category không bằng nhau:

```text
Manipulated Images: 32 ảnh
Natural Scenes: 31 ảnh
Social Scenes: 22 ảnh
Synthetic Images: 15 ảnh
```

Nếu dùng từng category riêng, nhóm có nhiều ảnh hơn có thể có lợi thế vì model nhận được nhiều thông tin hơn.

Thí nghiệm cân bằng xử lý vấn đề này bằng cách:

```text
Subset-20: mỗi category lấy 5 ảnh, tổng 20 ảnh.
Subset-40: mỗi category lấy 10 ảnh, tổng 40 ảnh.
Subset-60: mỗi category lấy 15 ảnh, tổng 60 ảnh.
```

Không có Subset-80 vì `Synthetic Images` chỉ có 15 ảnh. Nếu muốn cân bằng 4 category, mức cao nhất là:

```text
15 ảnh/category x 4 category = 60 ảnh
```

### 0.3. Baseline để so sánh

Baseline vẫn là kết quả 100 ảnh từ notebook/kết quả cũ:

```text
All-100
Accuracy = 0.8438
Sensitivity = 0.8463
Specificity = 0.8361
Precision = 0.8403
F1 = 0.8405
AUC = 0.9503
```

Baseline này không cân bằng theo category vì nó dùng tất cả 100 ảnh gốc.

---

## 1. File Mới Sau Khi Pull

### 1.1. Tài liệu và lệnh chạy

```text
BALANCED_STIMULUS_EXPERIMENT.md
message2.txt
run_balanced_stimulus_experiment.ps1
```

Ý nghĩa:

- `BALANCED_STIMULUS_EXPERIMENT.md`: mô tả ngắn thí nghiệm cân bằng 20/40/60 ảnh.
- `message2.txt`: yêu cầu thí nghiệm chi tiết, bằng tiếng Việt.
- `run_balanced_stimulus_experiment.ps1`: lệnh PowerShell chạy toàn bộ pipeline.

### 1.2. Code chính

```text
balanced_stimulus_experiment/
|-- __init__.py
|-- common.py
|-- prepare.py
|-- train_msnet.py
|-- run_fixed_protocol.py
`-- validate.py
```

Vai trò từng file:

- `common.py`: khai báo đường dẫn, danh sách số ảnh `20/40/60`, seed `42`, công thức ghép điểm `0.8/0.2`, tên fold và tên category.
- `prepare.py`: tạo danh sách ảnh cân bằng và cache dữ liệu 60 ảnh.
- `train_msnet.py`: huấn luyện MSNet cho 20/40/60 ảnh, mỗi mức chạy 4 fold.
- `run_fixed_protocol.py`: chạy phần hành vi mắt nhìn, ghép điểm MSNet + behavior, tính metric và viết báo cáo.
- `validate.py`: kiểm tra thí nghiệm có giữ đúng quy tắc không.

### 1.3. Output mới

```text
outputs/balanced_stimulus_experiment/
|-- manifests/
|   |-- stimulus_20.csv
|   |-- stimulus_40.csv
|   `-- stimulus_60.csv
|-- results_20/
|-- results_40/
|-- results_60/
|-- summary/
|   |-- aggregate_metrics.csv
|   |-- experiment_summary.md
|   |-- fold_metrics.csv
|   |-- fusion_auc_by_fold.csv
|   `-- metric_vs_stimulus_count.png
|-- protocol.json
`-- validation_report.json
```

Ý nghĩa:

- `manifests/`: danh sách ảnh được chọn cho từng mức 20/40/60.
- `results_20`, `results_40`, `results_60`: kết quả riêng cho từng mức số ảnh.
- `summary/`: bảng tổng hợp cuối cùng để đọc và đưa vào báo cáo.
- `protocol.json`: ghi lại cấu hình thí nghiệm.
- `validation_report.json`: kết quả kiểm tra tự động, hiện đang `PASS`.

---

## 2. Luồng Thí Nghiệm

### 2.1. Luồng tổng quát

```text
Category cache cũ
  |
  v
balanced_stimulus_experiment/prepare.py
  |
  v
tạo 3 manifest cân bằng: 20, 40, 60 ảnh
  |
  v
tạo master cache 60 ảnh
  |
  v
balanced_stimulus_experiment/train_msnet.py
  |
  v
huấn luyện MSNet cho 20, 40, 60 ảnh
  |
  v
balanced_stimulus_experiment/run_fixed_protocol.py
  |
  v
tạo Pupil + Scanpath behavior theo đúng ảnh đã chọn
  |
  v
ghép điểm 0.8 MSNet + 0.2 behavior
  |
  v
tính metric và so sánh với All-100 baseline
  |
  v
balanced_stimulus_experiment/validate.py
  |
  v
kiểm tra quy trình và ghi validation_report.json
```

### 2.2. Lệnh chạy toàn bộ

```powershell
.\run_balanced_stimulus_experiment.ps1
```

Lệnh này chạy lần lượt:

```text
1. prepare.py
2. train_msnet.py --stimulus-count all --fold all --epochs 50 --batch-size 8 --seed 42
3. run_fixed_protocol.py
4. validate.py
```

Nếu thiếu môi trường ảo `.venv`, script sẽ dừng và báo lỗi.

---

## 3. Bước 1 - Tạo Danh Sách Ảnh Cân Bằng

### 3.1. Đầu vào

`prepare.py` không đọc trực tiếp từ `feature_dict_RINet.npy` để tạo từng ảnh một. Nó dùng lại cache category đã có từ thí nghiệm trước:

```text
work/category_experiments/cache/manipulated/features.npy
work/category_experiments/cache/natural/features.npy
work/category_experiments/cache/social/features.npy
work/category_experiments/cache/synthetic/features.npy
```

Mỗi category cache đi kèm:

```text
manifest.json
subjects.csv
features.npy
```

### 3.2. Cách chọn ảnh

Mỗi category có danh sách ảnh riêng. Code làm:

```text
1. Sắp xếp tên ảnh trong từng category.
2. Trộn danh sách bằng seed 42.
3. Lấy ảnh theo vòng:
   manipulated -> natural -> social -> synthetic
   rồi lặp lại.
```

Cách lấy theo vòng giúp mọi prefix đều cân bằng.

Ví dụ đơn giản:

```text
Vòng 1:
  1 ảnh Manipulated
  1 ảnh Natural
  1 ảnh Social
  1 ảnh Synthetic

Vòng 2:
  thêm 1 ảnh mỗi category

...

Vòng 15:
  tổng 15 ảnh mỗi category = 60 ảnh
```

### 3.3. Vì sao gọi là tập lồng nhau?

Tập lồng nhau nghĩa là tập nhỏ nằm trong tập lớn.

Trong code:

```text
Subset-20 = 20 ảnh đầu của master order
Subset-40 = 40 ảnh đầu của master order
Subset-60 = 60 ảnh đầu của master order
```

Vì vậy:

```text
Subset-20 nằm trong Subset-40
Subset-40 nằm trong Subset-60
```

Ký hiệu:

```text
S20 subset S40 subset S60
```

Ý nghĩa thực tế:

```text
Khi tăng từ 20 lên 40 ảnh, ta chỉ bổ sung thêm ảnh.
Ta không thay toàn bộ danh sách ảnh.
```

### 3.4. File manifest được tạo

```text
outputs/balanced_stimulus_experiment/manifests/stimulus_20.csv
outputs/balanced_stimulus_experiment/manifests/stimulus_40.csv
outputs/balanced_stimulus_experiment/manifests/stimulus_60.csv
```

Mỗi file có 2 cột:

```text
IMAGE
CATEGORY
```

Yêu cầu đúng:

```text
stimulus_20.csv: 20 dòng, mỗi category 5 ảnh.
stimulus_40.csv: 40 dòng, mỗi category 10 ảnh.
stimulus_60.csv: 60 dòng, mỗi category 15 ảnh.
```

---

## 4. Bước 2 - Tạo Master Cache 60 Ảnh

### 4.1. Cache là gì?

Cache là file trung gian để lần sau đọc nhanh hơn. Ở đây cache chứa dữ liệu đã gom sẵn cho MSNet.

File được tạo:

```text
work/balanced_stimulus_experiment/features_60.npy
work/balanced_stimulus_experiment/subjects.csv
work/balanced_stimulus_experiment/master_manifest.json
```

### 4.2. Vì sao chỉ tạo `features_60.npy`?

Vì 20 và 40 là phần đầu của 60:

```text
features_60.npy chứa 60 ảnh.
Khi chạy 20 ảnh, code lấy 20 ảnh đầu.
Khi chạy 40 ảnh, code lấy 40 ảnh đầu.
Khi chạy 60 ảnh, code lấy đủ 60 ảnh.
```

Trong `common.py`, hàm `master_cache(stimulus_count)` trả về:

```text
features[:, :, :stimulus_count, :]
```

Nói dễ hiểu:

```text
Đã có một hộp chứa 60 ảnh theo thứ tự cố định.
Cần 20 thì lấy 20 ảnh đầu.
Cần 40 thì lấy 40 ảnh đầu.
Cần 60 thì lấy hết.
```

### 4.3. Kích thước cache

Cache có dạng:

```text
160 người x 1056 đặc trưng x 60 ảnh x 14 điểm nhìn
```

Tức là:

```text
(160, 1056, 60, 14)
```

Ý nghĩa:

- `160`: số người trong validation.
- `1056`: số đặc trưng ảnh lấy từ RINet.
- `60`: số ảnh tối đa trong thí nghiệm cân bằng.
- `14`: tối đa 14 điểm mắt nhìn trên mỗi ảnh, theo cách làm EMS/MSNet.

### 4.4. Nguồn dữ liệu của cache

`prepare.py` không tạo lại RINet. Nó lấy dữ liệu từ cache category cũ:

```text
work/category_experiments/cache/<category>/features.npy
```

Vì vậy điều kiện để chạy được `prepare.py` là trước đó phải có cache category.

Nếu thiếu cache category, cần chạy:

```powershell
.\run_category_experiments.ps1 -Category all -Epochs 50
```

Hoặc ít nhất phải chạy phần `prepare_cache.py` của thí nghiệm category để tạo các cache:

```powershell
.\.venv\Scripts\python.exe .\category_experiments\prepare_cache.py --category all
```

---

## 5. Bước 3 - Huấn Luyện MSNet Cho 20/40/60 Ảnh

### 5.1. Vì sao vẫn huấn luyện lại MSNet?

MSNet phụ thuộc vào số ảnh đầu vào.

Trong baseline 100 ảnh:

```text
MSNet nhận 100 ảnh.
```

Trong thí nghiệm mới:

```text
Subset-20: MSNet nhận 20 ảnh.
Subset-40: MSNet nhận 40 ảnh.
Subset-60: MSNet nhận 60 ảnh.
```

Trong code, model được tạo bằng:

```text
Model(in_c=1056, stimulus_count=stimulus_count)
```

Vì `stimulus_count` thay đổi, model cho 20/40/60 ảnh có kích thước nội bộ khác model 100 ảnh gốc.

### 5.2. Số lần huấn luyện

Có 3 mức số ảnh:

```text
20
40
60
```

Mỗi mức chạy 4 fold:

```text
Set_0
Set_1
Set_2
Set_3
```

Tổng cộng:

```text
3 mức x 4 fold = 12 lần huấn luyện MSNet
```

Mỗi lần huấn luyện mặc định:

```text
50 epochs
batch size = 8
seed = 42
```

`Epoch` nghĩa là một vòng model xem qua phần dữ liệu dùng để học. 50 epochs nghĩa là lặp quá trình học 50 lần.

### 5.3. Cách chia train/validation trong mỗi fold

Ví dụ với `Set_0`:

```text
Validation: Set_0, 40 người
Train: Set_1 + Set_2 + Set_3, 120 người
```

Tương tự:

```text
Set_1 làm validation -> train trên Set_0 + Set_2 + Set_3
Set_2 làm validation -> train trên Set_0 + Set_1 + Set_3
Set_3 làm validation -> train trên Set_0 + Set_1 + Set_2
```

Mục đích:

```text
Mỗi người trong 160 người được kiểm tra đúng một lần bằng model chưa học từ người đó.
```

### 5.4. File MSNet được tạo

Với 20 ảnh:

```text
outputs/balanced_stimulus_experiment/results_20/msnet/Set_0/best_model.pt
outputs/balanced_stimulus_experiment/results_20/msnet/Set_1/best_model.pt
outputs/balanced_stimulus_experiment/results_20/msnet/Set_2/best_model.pt
outputs/balanced_stimulus_experiment/results_20/msnet/Set_3/best_model.pt
```

Tương tự cho:

```text
outputs/balanced_stimulus_experiment/results_40/msnet/
outputs/balanced_stimulus_experiment/results_60/msnet/
```

Mỗi fold còn có:

```text
prob_val.csv
fold_metrics.json
training_history.csv
training_state.pt
completed.json
```

Ý nghĩa:

- `best_model.pt`: trọng số MSNet tốt nhất trong lần huấn luyện đó.
- `prob_val.csv`: điểm dự đoán MSNet cho 40 người validation của fold đó.
- `fold_metrics.json`: kết quả MSNet riêng của fold.
- `training_history.csv`: lịch sử từng epoch.
- `training_state.pt`: trạng thái để resume nếu chạy dở.
- `completed.json`: đánh dấu fold đã chạy xong.

---

## 6. Bước 4 - Chạy Behavioral Và Fusion Cố Định

### 6.1. Behavioral là gì?

Behavioral là phần dùng các con số mô tả hành vi mắt nhìn.

Trong thí nghiệm này chỉ dùng:

```text
Pupil + Scanpath
```

Không dùng:

```text
Duration
```

Giải thích:

- `Pupil`: đồng tử thay đổi ra sao.
- `Scanpath`: mắt di chuyển qua các vùng nào, dài/ngắn/lan rộng thế nào.
- `Duration`: nhìn lâu bao nhiêu; phần này không dùng trong fixed protocol mới.

### 6.2. Cách lọc behavioral theo manifest

Với mỗi mức 20/40/60, code đọc:

```text
improve_behavioral_final_results/stim_behavior_features.csv
```

Sau đó chỉ giữ các dòng có ảnh nằm trong manifest tương ứng.

Ví dụ với 20 ảnh:

```text
Chỉ giữ fixation/features của 20 ảnh trong stimulus_20.csv.
Không lấy thông tin từ 80 ảnh còn lại.
```

### 6.3. Cách học behavioral

Mỗi fold:

```text
Train: 3 fold còn lại
Validation: fold hiện tại
```

Model hành vi:

```text
HistGradientBoostingClassifier
max_iter = 300
learning_rate = 0.03
max_leaf_nodes = 15
l2_regularization = 0.01
random_state = 42
```

Nói dễ hiểu:

```text
Máy học từ 173 con số Pupil + Scanpath của mỗi người,
rồi tạo ra p_behavior, tức điểm nghiêng về SZ.
```

### 6.4. Ghép điểm

Sau khi có:

```text
p_msnet
p_behavior
```

code ghép:

```text
p_fusion = 0.8 * p_msnet + 0.2 * p_behavior
```

Không tìm lại alpha mới. Alpha luôn là:

```text
MSNet: 0.8
Behavior: 0.2
```

### 6.5. File fusion được tạo

Với từng mức số ảnh:

```text
outputs/balanced_stimulus_experiment/results_20/fold_metrics.csv
outputs/balanced_stimulus_experiment/results_20/feature_columns.json
outputs/balanced_stimulus_experiment/results_20/subject_predictions.csv
```

Tương tự cho:

```text
results_40/
results_60/
```

`subject_predictions.csv` có các cột:

```text
subject_id
subject_int
fold
stimulus_count
true_label
p_msnet
p_behavior
p_fusion
threshold
predicted_label
```

---

## 7. Bước 5 - Tổng Hợp Kết Quả

### 7.1. File tổng hợp

```text
outputs/balanced_stimulus_experiment/summary/aggregate_metrics.csv
outputs/balanced_stimulus_experiment/summary/fold_metrics.csv
outputs/balanced_stimulus_experiment/summary/fusion_auc_by_fold.csv
outputs/balanced_stimulus_experiment/summary/subject_predictions.csv
outputs/balanced_stimulus_experiment/summary/experiment_summary.md
outputs/balanced_stimulus_experiment/summary/metric_vs_stimulus_count.png
```

### 7.2. Bảng kết quả hiện có

```text
| Stimuli | Images/category | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC | Delta AUC |
| 20      | 5               | 0.7563   | 0.7693      | 0.7264      | 0.7267    | 0.7447 | 0.8367 | -0.1135 |
| 40      | 10              | 0.7562   | 0.8483      | 0.6576      | 0.7127    | 0.7722 | 0.8815 | -0.0688 |
| 60      | 15              | 0.8375   | 0.8952      | 0.7743      | 0.7965    | 0.8409 | 0.9170 | -0.0332 |
| 100     | unbalanced      | 0.8438   | 0.8463      | 0.8361      | 0.8403    | 0.8405 | 0.9503 | 0.0000 |
```

### 7.3. Cách đọc kết quả

Từ 20 lên 40 ảnh:

```text
AUC tăng từ 0.8367 lên 0.8815
Tăng +0.0447
```

Từ 40 lên 60 ảnh:

```text
AUC tăng từ 0.8815 lên 0.9170
Tăng +0.0356
```

So với baseline 100 ảnh:

```text
60 ảnh thấp hơn baseline 0.0332 AUC.
Accuracy 60 ảnh thấp hơn baseline 0.0062.
F1 60 ảnh gần như ngang baseline, hơn 0.0003.
Sensitivity 60 ảnh cao hơn baseline 0.0489.
Specificity 60 ảnh thấp hơn baseline 0.0618.
```

Nói dễ hiểu:

```text
60 ảnh cân bằng giữ được Accuracy và F1 gần baseline,
nhưng AUC vẫn thấp hơn baseline,
nhận SZ tốt hơn,
nhận HC kém hơn.
```

### 7.4. Kết quả AUC theo từng fold

```text
Set_0:
20 ảnh = 0.8737
40 ảnh = 0.8460
60 ảnh = 0.9040

Set_1:
20 ảnh = 0.7292
40 ảnh = 0.8906
60 ảnh = 0.8620

Set_2:
20 ảnh = 0.8450
40 ảnh = 0.8600
60 ảnh = 0.9450

Set_3:
20 ảnh = 0.8990
40 ảnh = 0.9293
60 ảnh = 0.9571
```

Điểm cần chú ý:

```text
60 ảnh tốt nhất trung bình,
nhưng không thắng mọi fold.
Ví dụ Set_1: 40 ảnh cao hơn 60 ảnh.
```

Vì vậy báo cáo nên nói:

```text
Trung bình 4 fold cho thấy 60 ảnh gần baseline nhất.
```

Không nên nói:

```text
60 ảnh luôn tốt hơn 40 ảnh trong mọi trường hợp.
```

---

## 8. Bước 6 - Kiểm Chứng Tự Động

### 8.1. Validator kiểm tra gì?

File:

```text
balanced_stimulus_experiment/validate.py
```

Nó kiểm tra:

- `stimulus_20.csv` có đúng 20 ảnh.
- `stimulus_40.csv` có đúng 40 ảnh.
- `stimulus_60.csv` có đúng 60 ảnh.
- Mỗi mức có đủ 4 category.
- Mỗi category có số ảnh bằng nhau trong từng mức.
- `S20` nằm trong `S40`.
- `S40` nằm trong `S60`.
- Cache master có shape `(160, 1056, 60, 14)`.
- Behavioral không lấy ảnh ngoài manifest.
- Mỗi mức có đủ 160 người.
- Mỗi fold có 40 người.
- Công thức fusion đúng tuyệt đối `0.8/0.2`.
- Điểm `p_msnet`, `p_behavior`, `p_fusion` nằm trong khoảng 0 đến 1.
- `predicted_label` khớp với threshold.
- Feature behavior có đúng 173 cột.
- Có cột pupil.
- Không có cột duration.
- Mỗi mức có đủ 4 fold MSNet hoàn tất.

### 8.2. Kết quả kiểm chứng hiện có

File:

```text
outputs/balanced_stimulus_experiment/validation_report.json
```

Hiện báo:

```text
status = PASS
nested = S20 subset S40 subset S60
cache_shape = [160, 1056, 60, 14]
```

Từng mức:

```text
20 ảnh: 5 ảnh/category, 160 người, 4 fold MSNet hoàn tất.
40 ảnh: 10 ảnh/category, 160 người, 4 fold MSNet hoàn tất.
60 ảnh: 15 ảnh/category, 160 người, 4 fold MSNet hoàn tất.
```

---

## 9. Những Gì Thí Nghiệm Này Trả Lời Được

### 9.1. Câu hỏi trả lời được

Thí nghiệm trả lời tốt các câu hỏi:

```text
Nếu giảm số ảnh còn 20/40/60 nhưng vẫn cân bằng 4 category,
kết quả phân biệt SZ/HC giảm bao nhiêu so với 100 ảnh?
```

```text
Mức 60 ảnh có gần baseline 100 ảnh không?
```

```text
Khi tăng ảnh từ 20 lên 40, từ 40 lên 60, AUC tăng bao nhiêu?
```

```text
Giảm số ảnh ảnh hưởng khác nhau thế nào đến nhận SZ và nhận HC?
```

### 9.2. Câu hỏi không trả lời trực tiếp

Thí nghiệm này không trả lời trực tiếp:

```text
Natural / Social / Synthetic / Manipulated category nào tốt nhất?
```

Vì mỗi subset đều chứa cả 4 category với số ảnh bằng nhau.

Nó cũng không trả lời:

```text
Nếu không huấn luyện lại MSNet thì từng nhóm ảnh đóng góp bao nhiêu?
```

Muốn trả lời câu đó cần làm phân tích mask/keep-one như đã mô tả trong `plan_1.md`.

---

## 10. Diễn Giải Khoa Học Nên Viết

### 10.1. Câu kết luận phù hợp

Có thể viết:

```text
Khi giữ cấu hình fusion baseline và dùng các tập ảnh cân bằng giữa 4 category,
hiệu năng tăng dần khi số ảnh tăng từ 20 lên 60.
Tập 60 ảnh đạt AUC 0.9170, thấp hơn baseline 100 ảnh 0.0332,
nhưng có Accuracy và F1 rất gần baseline.
```

Có thể viết thêm:

```text
Kết quả cho thấy có thể giảm từ 100 ảnh xuống 60 ảnh trong điều kiện cân bằng category
mà vẫn giữ được phần lớn hiệu năng validation, dù AUC vẫn thấp hơn baseline.
```

### 10.2. Câu không nên viết

Không nên viết:

```text
60 ảnh tốt tương đương hoàn toàn 100 ảnh.
```

Vì AUC vẫn thấp hơn `0.0332`.

Không nên viết:

```text
Thí nghiệm này chứng minh category nào tốt nhất.
```

Vì đây là thí nghiệm số lượng ảnh cân bằng, không phải thí nghiệm so sánh từng category.

Không nên viết:

```text
Đây là kết quả test set.
```

Vì đây là four-fold validation trên 160 người.

Không nên viết:

```text
Kết quả chắc chắn ổn định với mọi cách chọn ảnh.
```

Vì hiện mới dùng một seed chọn ảnh là `42`.

---

## 11. Các Hạn Chế Cần Ghi Rõ

### 11.1. Chỉ có một seed chọn ảnh

Code hiện dùng:

```text
SEED = 42
```

Nghĩa là chỉ có một cách chọn ảnh 20/40/60.

Nếu chọn ảnh khác, kết quả có thể thay đổi.

Để kết luận mạnh hơn, nên chạy nhiều seed:

```text
seed 1
seed 2
seed 3
seed 4
seed 5
...
```

Sau đó báo cáo trung bình và độ dao động.

### 11.2. MSNet vẫn được huấn luyện lại

Vì số ảnh đầu vào thay đổi, MSNet phải được huấn luyện theo từng mức:

```text
20 ảnh
40 ảnh
60 ảnh
```

Điều này hợp lý về kỹ thuật, nhưng cần ghi rõ:

```text
Kết quả phản ánh pipeline được huấn luyện cho từng số lượng ảnh,
không phải dùng nguyên checkpoint 100 ảnh của tác giả.
```

### 11.3. Baseline 100 ảnh không cân bằng category

Baseline dùng tất cả ảnh:

```text
32 Manipulated
31 Natural
22 Social
15 Synthetic
```

Trong khi 60 ảnh cân bằng là:

```text
15 Manipulated
15 Natural
15 Social
15 Synthetic
```

Vì vậy khi so sánh 60 với 100, có hai khác biệt:

```text
số ảnh ít hơn
phân bố category cân bằng hơn
```

Kết quả không chỉ là tác động của "ít ảnh hơn".

### 11.4. Cần cache category cũ

`prepare.py` dùng:

```text
work/category_experiments/cache/<category>/features.npy
```

Nếu người khác clone repo mà chưa có cache này, họ không chạy được ngay.

Cần tài liệu hóa bước chuẩn bị:

```powershell
.\.venv\Scripts\python.exe .\category_experiments\prepare_cache.py --category all
```

---

## 12. Đề Xuất Mở Rộng Nếu Muốn Kết Luận Mạnh Hơn

### 12.1. Chạy nhiều seed

Mục tiêu:

```text
Kiểm tra kết quả có phụ thuộc vào một lần chọn ảnh seed 42 hay không.
```

Thiết kế:

```text
seed = 1, 2, 3, 4, 5
mỗi seed tạo S20/S40/S60 mới
chạy lại pipeline
báo cáo trung bình và độ dao động AUC
```

Output nên có:

```text
outputs/balanced_stimulus_experiment_multiseed/summary/aggregate_by_seed.csv
outputs/balanced_stimulus_experiment_multiseed/summary/aggregate_mean_std.csv
```

### 12.2. Thêm điều kiện 15 ảnh/category theo từng category riêng

Mục tiêu:

```text
Tách ảnh hưởng số lượng ảnh khỏi ảnh hưởng category.
```

Ví dụ:

```text
Natural lấy 15 ảnh
Social lấy 15 ảnh
Synthetic lấy 15 ảnh
Manipulated lấy 15 ảnh
```

Mức này giúp so sánh category công bằng hơn vì mỗi category có cùng số ảnh.

### 12.3. Phân tích đóng góp không huấn luyện lại MSNet

Mục tiêu:

```text
Dùng nguyên model 100 ảnh của tác giả,
che từng nhóm ảnh để xem nhóm nào đóng góp nhiều nhất.
```

Đây là thí nghiệm kiểm soát tốt nếu bị phản biện rằng MSNet huấn luyện lại làm kết quả lệch.

---

## 13. Checklist Xác Minh

- [ ] `BALANCED_STIMULUS_EXPERIMENT.md` tồn tại và mô tả đúng 20/40/60 ảnh.
- [ ] `message2.txt` tồn tại và mô tả đúng yêu cầu giữ cấu hình baseline.
- [ ] `run_balanced_stimulus_experiment.ps1` chạy đủ 4 bước: prepare, train, run fixed protocol, validate.
- [ ] `outputs/balanced_stimulus_experiment/manifests/stimulus_20.csv` có 20 ảnh.
- [ ] `outputs/balanced_stimulus_experiment/manifests/stimulus_40.csv` có 40 ảnh.
- [ ] `outputs/balanced_stimulus_experiment/manifests/stimulus_60.csv` có 60 ảnh.
- [ ] Mỗi manifest có đủ 4 category.
- [ ] Mỗi category trong `stimulus_20.csv` có 5 ảnh.
- [ ] Mỗi category trong `stimulus_40.csv` có 10 ảnh.
- [ ] Mỗi category trong `stimulus_60.csv` có 15 ảnh.
- [ ] `S20` nằm trong `S40`.
- [ ] `S40` nằm trong `S60`.
- [ ] Cache master có shape `(160, 1056, 60, 14)`.
- [ ] Mỗi mức 20/40/60 có đủ 4 fold MSNet.
- [ ] Mỗi mức 20/40/60 có đủ 160 prediction.
- [ ] Mỗi fold có 40 người.
- [ ] Feature behavior là `Pupil + Scanpath`.
- [ ] Số feature behavior là 173.
- [ ] Không có feature `duration_`.
- [ ] Fusion đúng công thức `0.8 * p_msnet + 0.2 * p_behavior`.
- [ ] `validation_report.json` có `status = PASS`.
- [ ] Báo cáo cuối ghi rõ đây là validation, không phải held-out test.
- [ ] Báo cáo cuối ghi rõ đây là single-seed experiment.

---

## 14. Kết Luận Ngắn Cho Người Đọc

Thí nghiệm mới trong repo là thí nghiệm giảm số lượng ảnh nhưng giữ cân bằng 4 category.

Nó tạo:

```text
20 ảnh = 5 ảnh/category
40 ảnh = 10 ảnh/category
60 ảnh = 15 ảnh/category
```

Rồi với mỗi mức, nó:

```text
huấn luyện MSNet 4 fold
tạo điểm hành vi Pupil + Scanpath
ghép điểm 0.8 / 0.2
tính metric
so sánh với baseline 100 ảnh
```

Kết quả hiện có:

```text
20 ảnh: AUC 0.8367
40 ảnh: AUC 0.8815
60 ảnh: AUC 0.9170
100 ảnh: AUC 0.9503
```

Kết luận phù hợp:

```text
Trong một lần chọn ảnh bằng seed 42, 60 ảnh cân bằng là mức gần baseline 100 ảnh nhất.
Nó giữ Accuracy và F1 gần baseline, nhưng AUC vẫn thấp hơn baseline.
```
