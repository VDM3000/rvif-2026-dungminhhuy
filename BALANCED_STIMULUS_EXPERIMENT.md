# Balanced EMS stimulus-count experiment

This experiment follows `message2.txt` and changes only the selected stimulus
list. Three nested, category-balanced conditions are generated with seed 42:

| Condition | Images/category | Total |
|---|---:|---:|
| Subset-20 | 5 | 20 |
| Subset-40 | 10 | 40 |
| Subset-60 | 15 | 60 |

Every condition uses four-fold MSNet training followed by fixed Pupil+Scanpath
HistGradientBoosting and `p_fusion = 0.8*p_msnet + 0.2*p_behavior`. The existing
All-100 Kaggle AUC 0.9503 remains the unchanged pretrained reference.

Run or resume the complete pipeline:

```powershell
.\run_balanced_stimulus_experiment.ps1
```

Completed MSNet folds are skipped automatically. Local cache and subject-level
predictions are intentionally excluded from Git. Aggregate review outputs are in
`outputs/balanced_stimulus_experiment/summary`.

This is a single-seed experiment; it does not quantify uncertainty caused by
choosing a different random set of stimuli.
