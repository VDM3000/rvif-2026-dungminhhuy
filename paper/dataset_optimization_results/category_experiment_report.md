# Fixed original protocol results

Cấu hình được giữ cố định cho mọi category: MSNet + Pupil/Scanpath + HGB, `p_fusion = 0.8*p_msnet + 0.2*p_behavior`.

| Stimulus set | # Images | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| All | 100 | 0.8438 | 0.8463 | 0.8361 | 0.8405 | 0.9503 |
| Manipulated Images | 32 | 0.7937 | 0.7736 | 0.8000 | 0.7775 | 0.9100 |
| Natural Scenes | 31 | 0.8125 | 0.8608 | 0.7465 | 0.8088 | 0.9129 |
| Social Scenes | 22 | 0.7937 | 0.8182 | 0.7681 | 0.7928 | 0.8887 |
| Synthetic Images | 15 | 0.7250 | 0.7145 | 0.7264 | 0.7147 | 0.8434 |

- AUC cao nhất: Natural Scenes (0.9129).
- Sensitivity cao nhất (nhận SZ): Natural Scenes (0.8608).
- Specificity cao nhất (nhận HC): Manipulated Images (0.8000).
- AUC ổn định nhất qua fold: Manipulated Images (SD=0.0392).

Các kết luận chỉ áp dụng cho EMS, 160 subjects và four-fold validation trong cấu hình trên.
