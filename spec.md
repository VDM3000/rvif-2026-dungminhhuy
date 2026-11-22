# Đặc tả bốn thí nghiệm bổ sung cho BAMF

**Ngày lập:** 2026-11-22. **Trạng thái:** đã chạy và kiểm tra bốn thí nghiệm, cập nhật `main.tex` với phần thay đổi màu xanh và biên dịch bản PDF kiểm tra. **Theo dõi:** `EXPERIMENT_CHECKPOINTS.md`.

## 1. Mục tiêu và phạm vi

Thực hiện bốn đề xuất của GVHD: chạy thêm seed, so sánh Early Fusion với Late Fusion, kiểm tra/giải thích đồ thị α, và đánh giá Late Fusion theo cách chia 2/1/1. Bản thảo gốc ở `../2026_RVIF_HuyMinhDung_Final/main.tex`; bản bàn giao trong repository hiện ở `paper/main.tex` và `paper/main.pdf`. Không sửa số liệu hoặc lời văn trong bài trước khi có kết quả đã kiểm tra.

**Nguồn hiện có:** `notebooks/windows.ipynb` đã tạo lại kết quả BAMF chính; `ems_overrides/MSNet.py` định nghĩa MSNet; `EMS/Train_Valid.xlsx` chứa 160 người ở bốn fold; `feature_dict_RINet.npy`, `EMS_repo/`, `.venv/` và `model_parameters/` hiện có trên máy. `outputs/reviewer_experiments/baseline_2026-11-22_seed42/` giữ bản sao kết quả và manifest của lần tái tạo. Không sử dụng 48 người test EMS để tính metric vì không có nhãn.

**Kết quả lịch sử để đối chiếu:** Late Fusion Pupil + Scanpath (173 chiều), HGB, trọng số MSNet α=0,8, four-fold mean AUC 95,03%; MSNet 89,72%. Kết quả mới phải ghi vào `outputs/reviewer_experiments/<run_id>/`, không ghi đè bảng lịch sử.

## 2. Quy tắc chung cho mọi lần chạy

1. Một người chỉ thuộc một fold. Danh sách fold lấy từ `EMS/Train_Valid.xlsx`, không tạo chia ngẫu nhiên mới.
2. Trước khi chạy, lưu commit mã chính và EMS, hash notebook, feature dictionary và checkpoint, phiên bản thư viện, seed, cấu hình, thời gian chạy và tên thư mục đầu ra trong `manifest.json`.
3. Đầu ra theo người có `subject_id`, `fold`, nhãn, score MSNet, score behavioral/early fusion, nhãn dự đoán và nguồn checkpoint. Kiểm tra đủ 160 người không trùng khi tổng hợp bốn fold.
4. Dữ liệu của fold đánh giá không tham gia huấn luyện, chọn α, chọn epoch, chọn ngưỡng theo nhãn, chọn đặc trưng hoặc chọn phương pháp thắng. Nếu tính Otsu từ score của fold đánh giá như protocol lịch sử, ghi rõ đây là bước phụ thuộc phân bố score của fold đó; AUC không phụ thuộc ngưỡng.
5. Báo AUC, accuracy, sensitivity, specificity, F1 từng fold và trung bình bốn fold. So sánh cùng người, cùng fold và cùng cách tính metric. Không dùng một số cao hơn như bằng chứng tự thân rằng protocol tốt hơn.
6. Seed kiểm soát các nguồn ngẫu nhiên trong Python, NumPy, PyTorch và mô hình hành vi; nếu thành phần không thay đổi theo seed, ghi rõ. Không điều chỉnh các tham số sau khi xem fold đánh giá mà vẫn gọi kết quả là xác nhận độc lập.

## 3. Thí nghiệm S — ba seed

**Câu hỏi:** kết quả hiện tại có thay đổi nhiều khi đổi seed hay không?

- Seed cố định trước khi chạy: `42`, `17`, `2026`. Dùng cùng bốn fold, cùng tập Pupil + Scanpath, HGB và cùng quy tắc α như kết quả lịch sử. Không chọn seed cho kết quả tốt nhất để đưa lên bài.
- Trước tiên tái chạy seed 42 và đối chiếu các bảng lịch sử. Ghi chính xác thành phần nào dùng seed: HGB, các bước có sampling và huấn luyện MSNet nếu có. MSNet pretrained và tập người cố định sẽ không tạo ba lần huấn luyện MSNet khác nhau trong kết quả chính.
- Mỗi seed ghi dự đoán theo người, kết quả từng fold, alpha từng fold và `seed_metrics.csv`. Bảng tổng hợp cho mỗi metric gồm ba giá trị, mean và **sample variance** với mẫu số `3−1`; thêm độ lệch chuẩn nếu hữu ích.
- Ba seed chỉ cho thấy độ dao động giữa ba lần chạy. Mean/variance của ba seed **không phải khoảng tin cậy 95%**, không thay thế kiểm định mức tăng BAMF–MSNet và không đo được độ biến thiên của quần thể. Nếu tất cả kết quả giống hệt, báo điều đó thay vì tuyên bố độ bền vững đã được xác nhận.

**Đạt khi:** có ba run manifests, 12 hàng seed×fold, ba giá trị của mỗi metric và phép tính mean/variance tái lập được; seed 42 được đối chiếu với số liệu bài báo, mọi sai khác có giải thích.

## 4. Thí nghiệm F — Early Fusion ở cấp độ đặc trưng

**Câu hỏi:** ghép đặc trưng trước khi phân loại có tốt hơn cách ghép hai score hiện tại không?

- MSNet tạo vector ngay trước `self.head`: trong cấu hình 100 ảnh, `self.norm(cluster_center4)` có **56 chiều** theo kiến trúc hiện tại. Cần thêm cách xuất vector này mà vẫn giữ đường dự đoán MSNet cũ; kiểm tra bằng tensor thật và checkpoint trước khi huấn luyện.
- Behavioral vector là 173 cột Pupil + Scanpath trong `feature_sets_clean.json`, được tạo từ bản ghi của cùng người. Kiểm tra tên, thứ tự, số chiều, giá trị thiếu và ID khi ghép với vector MSNet.
- Vector ghép dự kiến có `56 + 173 = 229` chiều. `head` gốc của MSNet nhận 56 chiều và gồm tuyến tính + sigmoid, nên không thể đưa thẳng 229 chiều vào nó. Tạo một classification head/MLP mới nhận 229 chiều; ghi cụ thể kiến trúc, regularization, số epoch và quy tắc chọn checkpoint trước khi chạy.
- Phương án chính để so sánh: dùng encoder MSNet được huấn luyện **chỉ trên 80 người train của cùng lượt 2/1/1**, giữ cố định encoder này khi huấn luyện head mới; imputation/scaling của 173 chiều fit trên người huấn luyện. Không dùng checkpoint pretrained ba-fold của kết quả lịch sử ở đây vì nó đã học nhãn của fold chọn α. Nếu fine-tune MSNet, đó là thí nghiệm khác, cần checkpoint và chi phí riêng.
- Đánh giá trên cùng fold và cùng người với Late Fusion. Để so sánh độc lập hơn, dùng phân chia 2/1/1 của Mục 6 cho **cả hai** phương pháp; nếu chỉ chạy four-fold lịch sử, gắn nhãn là so sánh phát triển nội bộ. Early Fusion không có α kết hợp hai score.
- Chọn epoch và mọi tham số của head bằng dữ liệu huấn luyện/chọn mô hình, không bằng fold cuối. Báo AUC và các metric từng fold; lưu prediction theo người và checkpoint head.

**Đạt khi:** kiểm tra được shape `(N,56)`, `(N,173)`, `(N,229)`, không trùng/mất subject, head huấn luyện được, có bốn fold kết quả và bảng so sánh cùng protocol với Late Fusion.

## 5. Thí nghiệm A — đồ thị α

**Câu hỏi:** đồ thị hiện tại thể hiện điều gì và có thể kết luận đến đâu?

- Xác minh file dữ liệu sinh đồ thị và phép tính: trục ngang là α từ 0 đến 1 bước 0,05; trục dọc là mean AUC trên bốn validation folds của cấu hình Pupil + Scanpath + HGB. Đối chiếu các điểm α=0, α=0,8 và α=1 với bảng gốc.
- Cụm “không có đoạn âm” phải xác định rõ đang nói **AUC không âm** hay **mức cải thiện so với MSNet không âm**. AUC vốn luôn nằm trong `[0,1]`, nên việc đường AUC không xuống dưới 0 không chứng minh fusion có lợi. Muốn khẳng định có lợi ở các α đã thử, tính `AUC_fusion(α) − AUC_MSNet` và kiểm tra từng điểm/từng fold; có điểm âm thì báo trung thực.
- Đồ thị hiện tại là phân tích trên validation set. 48 người test EMS không có nhãn, do đó kiểm chứng độc lập chỉ là hướng tương lai. Không gọi đường cong α là chứng minh khả năng áp dụng sang dữ liệu khác.
- Nếu có kết quả 2/1/1, vẽ riêng đường α trên fold chọn α và đánh dấu α được chọn; không dùng fold đánh giá để chọn đỉnh đồ thị.

**Đạt khi:** có bảng α và hình nguồn tái lập được, kiểm tra số liệu/ý nghĩa của “đoạn âm”, và đoạn nhận xét đúng loại dữ liệu được dùng.

## 6. Thí nghiệm L — Late Fusion với chia 2/1/1

**Câu hỏi:** Late Fusion còn hiệu quả thế nào khi fold chọn α và fold đánh giá là hai nhóm người khác nhau?

- Bốn lượt được cố định: fold đánh giá lần lượt `Set_0`, `Set_1`, `Set_2`, `Set_3`; fold chọn α lần lượt là fold kế tiếp theo vòng (`Set_1`, `Set_2`, `Set_3`, `Set_0`); hai fold còn lại để huấn luyện. Mỗi lượt có 80 train, 40 chọn α, 40 đánh giá.
- Cố định trước behavioral feature set `P+S` và HGB theo cấu hình đã công bố. Nếu muốn chọn lại trong 21 cấu hình, cần một tầng lựa chọn nằm trong 80 người train; không được chọn bằng fold đánh giá.
- Huấn luyện behavioral model trên 80 người; dự đoán 40 người chọn α, dùng nhãn của họ để chọn α trên grid 0–1 bước 0,05; khóa α; dự đoán 40 người cuối đúng một lần.
- **Lưu ý MSNet:** checkpoint pretrained `Set_i` của protocol cũ được huấn luyện trên ba fold ngoài `Set_i`, bao gồm fold đang dùng để chọn α trong lượt 2/1/1. Muốn tách hoàn toàn việc huấn luyện và chọn α, phải huấn luyện MSNet chỉ trên 80 người train cho mỗi lượt, với quy tắc chọn epoch/checkpoint không đọc fold đánh giá; hoặc chứng minh một nguồn checkpoint khác đáp ứng điều kiện này. Không gọi việc dùng lại checkpoint cũ là đánh giá 2/1/1 cách ly hoàn toàn.
- Ngưỡng cho accuracy/F1 được chọn từ 80 train hoặc 40 chọn α, rồi khóa trước khi chấm 40 người cuối. AUC không cần ngưỡng. Ghi từng lượt: các fold train/α/evaluation, checkpoint MSNet, α chọn, metric và dự đoán theo người.
- Tổng hợp kết quả bốn lượt và đối chiếu **có nhãn protocol** với 95,03% lịch sử; khác biệt có thể do ít dữ liệu huấn luyện hơn, thay checkpoint và thay cách chọn α. Không diễn giải chênh lệch như tác động riêng của α.

**Đạt khi:** bốn bảng 40 người, tổng 160 người duy nhất; không có fold cuối trong fitting/chọn tham số/checkpoint; có provenance và bảng so sánh protocol rõ ràng.

## 7. Thứ tự thực hiện và giới hạn

1. Khôi phục môi trường và tái tạo seed 42 trước. Lưu một bản frozen của kết quả lịch sử.
2. Chạy hai seed bổ sung và phân tích α từ dự đoán đã kiểm tra.
3. Khóa protocol 2/1/1 và chi phí huấn luyện MSNet; chạy Late Fusion theo protocol đó.
4. Chạy Early Fusion trên cùng protocol nếu có đủ checkpoint/đặc trưng để so sánh công bằng.
5. Chỉ sau khi kiểm tra output mới cập nhật `main.tex`, hình và bảng. Mọi đoạn thêm hoặc sửa so với `main.tex` có SHA-256 lưu trong manifest baseline phải được tô màu xanh trong LaTeX; kiểm tra màu trên PDF. Ba seed không được mô tả là CI; 2/1/1 vẫn là internal validation trên EMS, không phải external test.

Những lựa chọn phải chốt trước khi chạy được ghi trong checkpoint. Không tự động sửa bài báo hoặc công bố con số mới chỉ dựa trên đặc tả này.
