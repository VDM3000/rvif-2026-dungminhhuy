# RVIF 2026 — EMS stimulus-category experiments

This repository evaluates which EMS stimulus category provides the most useful
eye-movement information for four-fold SZ/HC discrimination. It accompanies the
fixed protocol described in [`message.txt`](message.txt).

## Fixed research protocol

All four categories use the same configuration:

```text
MSNet + Pupil/Scanpath + HistGradientBoosting
p_fusion = 0.8 * p_msnet + 0.2 * p_behavior
```

Only the stimulus subset changes. Subject membership, Set_0–Set_3 folds,
preprocessing, classifier settings, thresholding and metrics remain fixed.

| Stimulus set | Images | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| All-100 reference | 100 | 0.8438 | 0.8463 | 0.8361 | 0.8405 | 0.9503 |
| Manipulated Images | 32 | 0.7938 | 0.7736 | **0.8000** | 0.7775 | 0.9100 |
| Natural Scenes | 31 | **0.8125** | **0.8608** | 0.7465 | **0.8088** | **0.9129** |
| Social Scenes | 22 | 0.7938 | 0.8182 | 0.7681 | 0.7928 | 0.8887 |
| Synthetic Images | 15 | 0.7250 | 0.7145 | 0.7264 | 0.7147 | 0.8434 |

Natural Scenes has the highest observed category AUC and sensitivity;
Manipulated Images has the highest specificity and lowest fold-to-fold AUC SD.
These are four-fold validation observations on EMS, not held-out test results or
claims that a stimulus category is a schizophrenia biomarker.

See the [fixed-protocol report](outputs/fixed_original_protocol/summary/RESULTS_FIXED_PROTOCOL.md),
[fold AUC table](outputs/fixed_original_protocol/summary/auc_by_fold.csv), and
[validation report](outputs/fixed_original_protocol/validation_report.json).

## Repository layout

- `category_experiments/`: category cache, MSNet training, fixed fusion and validation code.
- `ems_overrides/`: Windows/category-compatible replacements for the EMS submodule.
- `outputs/fixed_original_protocol/`: aggregate metrics safe for public review.
- `outputs/category_experiments/summary/`: exploratory category-optimized analysis.
- `notebooks/original_kaggle.ipynb`: sanitized original Kaggle experiment notebook.
- `notebooks/windows.ipynb`: sanitized Windows-adapted notebook.

## Data and model assets

EMS data, RINet feature dictionaries, pretrained weights, caches, checkpoints and
subject-level predictions are intentionally excluded. Reviewers must obtain these
assets from their authorized source and place them locally as documented in
[`README_WINDOWS.md`](README_WINDOWS.md).

Expected local paths include `EMS/`, `feature_dict_RINet.npy`, and
`model_parameters/`. Do not commit these assets.

## Setup and reproduction

Clone with the EMS submodule and apply the local overrides:

```powershell
git clone --recurse-submodules https://github.com/VDM3000/rvif-2026-dungminhhuy.git
cd rvif-2026-dungminhhuy
.\apply_ems_overrides.ps1
.\setup_windows.ps1
```

After authorized data/model assets are present, run the fixed experiment:

```powershell
.\run_fixed_protocol.ps1
```

For methodological details, see [`FIXED_PROTOCOL.md`](FIXED_PROTOCOL.md).
The nested 20/40/60 category-balanced stimulus-count experiment is documented in
[`BALANCED_STIMULUS_EXPERIMENT.md`](BALANCED_STIMULUS_EXPERIMENT.md), with aggregate
results under `outputs/balanced_stimulus_experiment/summary`.

## Important distinction

`outputs/fixed_original_protocol` is the primary result. The older
`outputs/category_experiments/summary` directory is exploratory: it optimized
feature/model/alpha separately per category and must not be presented as the
fixed original protocol.
