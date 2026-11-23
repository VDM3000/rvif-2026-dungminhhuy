# Hai phép chạy 2/1/1 và kế hoạch tái lập

## Quy tắc chung

Mỗi lượt xoay bốn fold theo `reviewer_experiments/protocol_211.py`: 80 người ở hai fold để học MSNet và 21 mô hình hành vi, 40 người ở fold kế tiếp để lựa chọn, 40 người ở fold cuối chỉ để đánh giá. `Set_0` đến `Set_3` lần lượt giữ vai trò đánh giá. Bộ dữ liệu test 48 người EMS không có nhãn nên không tính metric ở đây. Mọi output mới nằm trong thư mục riêng, không sửa kết quả lịch sử.

## Variant A — fixed P+S/HGB

Giữ đúng nhóm Pupil+Scanpath và HGB đã chọn ở `main.tex`. Fit nhánh này trên 80 người; 40 người giữa chỉ chọn α và ngưỡng; chấm 40 người cuối. Chạy lại đường mới rồi đối chiếu dự đoán theo người với `late_211_2026-11-22/`.

## Variant B — joint selection on middle fold

Fit đủ 7 nhóm đặc trưng × LR/RF/HGB trên cùng 80 người. Trên đúng 40 người giữa, thử từng mô hình cùng 21 giá trị α. Xếp 441 cặp bằng AUC, sau đó F1, accuracy, rồi tên cấu hình và α để giải hòa; làm tròn ba metric đến 12 chữ số thập phân trước khi so để tránh sai khác số thực không có ý nghĩa. Khóa cả nhóm đặc trưng, mô hình, α và ngưỡng trước khi chấm 40 người cuối. Không sử dụng nhãn fold cuối để lựa chọn. Đây là quy trình khác với `select_211.py` cũ (chọn mô hình bằng phép chia nội bộ trong 80 người); giữ kết quả cũ dưới tên `inner-selected`, không gọi nó là Variant B.

## Cách chứng minh và bàn giao

- Viết một CLI Python dùng trên Windows/Linux/macOS, nhận đường dẫn đầu vào/output tường minh và có lệnh kiểm tra tài sản đầu vào. Giữ `EMS_repo/` là submodule ghim commit; không sửa upstream trong clone khi tái lập.
- Test việc Variant A khớp kết quả cố định đã lưu, 441 ứng viên chỉ đọc 40 người giữa, và fold cuối không tham gia chọn. Lưu manifest trước chạy, model ML, bảng ứng viên, dự đoán và metric của 40 người ở giữa/cuối mỗi lượt.
- Chạy đủ bốn lượt hai variant và tổng hợp 160 ID duy nhất mỗi variant. So với MSNet **2/1/1 huấn luyện trên cùng 80 người**, không lấy MSNet pretrained 3/1 làm baseline trực tiếp.
- Cập nhật `spec.md`, `EXPERIMENT_CHECKPOINTS.md`, README, bản đồ file, bài báo và PDF sau khi kết quả được kiểm tra. Phân biệt ba phép chạy: fixed, joint trên 40 người giữa, inner-selected trong 80 người.

## Giới hạn tái lập

Git clone chỉ lấy mã. EMS, `feature_dict_RINet.npy` và bốn checkpoint pretrained 3/1 không nằm trong Git; cần cung cấp riêng theo quyền sử dụng. Bốn checkpoint MSNet 2/1/1 có thể huấn luyện lại từ EMS và RINet cache theo lệnh công bố. Mã cần nêu checksum, phiên bản gói và đường dẫn rõ ràng, tránh đường dẫn ổ đĩa Windows của tác giả.
