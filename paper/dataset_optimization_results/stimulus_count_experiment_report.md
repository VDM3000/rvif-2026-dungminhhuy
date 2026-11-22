# Balanced stimulus-count experiment

Fixed configuration: MSNet + Pupil/Scanpath + HGB, `alpha_msnet=0.8`.
All-100 is the unchanged Kaggle pretrained four-fold reference.

| Stimuli | Images/category | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC | Delta AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5 | 0.7563 | 0.7693 | 0.7264 | 0.7267 | 0.7447 | 0.8367 | -0.1135 |
| 40 | 10 | 0.7562 | 0.8483 | 0.6576 | 0.7127 | 0.7722 | 0.8815 | -0.0688 |
| 60 | 15 | 0.8375 | 0.8952 | 0.7743 | 0.7965 | 0.8409 | 0.9170 | -0.0332 |
| 100 | unbalanced | 0.8438 | 0.8463 | 0.8361 | 0.8403 | 0.8405 | 0.9503 | 0.0000 |

Closest subset to baseline by absolute AUC delta: 60 stimuli (AUC=0.9170, delta=-0.0332).

- AUC gain from 20 to 40 stimuli: +0.0447.
- AUC gain from 40 to 60 stimuli: +0.0356.
- At 60 stimuli, accuracy differs from baseline by -0.0062 and F1 by +0.0003.
- At 60 stimuli, sensitivity differs by +0.0489, while specificity differs by -0.0618.

This single seed does not quantify uncertainty from stimulus sampling.
