# BAMF revision for supervisor review

Upload the contents of this folder to Overleaf and compile main.tex with pdfLaTeX.

This revision extends the behavior-aware MSNet paper with two stimulus-set design experiments while keeping the selected configuration fixed:

- behavioral features: Pupil + Scanpath (173 dimensions)
- classifier: HistGradientBoosting
- fusion: 0.8 * p_MSNet + 0.2 * p_behavioral
- evaluation: the official four subject folds with fold-specific Otsu thresholds

Added analyses:

1. Category-specific models for Manipulated Images (32), Natural Scenes (31), Social Scenes (22), and Synthetic Images (15).
2. Nested, category-balanced subsets of 20, 40, and 60 images (5, 10, and 15 images per category; seed 42).

The source result tables, fold AUC values, experiment reports, and validator output are preserved in dataset_optimization_results/ for audit and co-author review.

Use [main.pdf](main.pdf) and [Supplementary Figure S1](supplement/alpha_211.pdf) for supervisor review. The paper includes the three-seed analysis, the historical fusion-weight analysis, and the 2/1/1 comparison. The substantive revisions are in blue. Tables III and VI use column widths that fit the two-column layout. Figure S1 is a separate attachment to keep the main paper at seven pages.

The [retrospective reproducibility record](../outputs/reviewer_experiments/audit_2026-11-22_mac/reproducibility_record.json) documents the saved configurations, checkpoint hashes, and details that could be confirmed after training. The [audit report](../outputs/reviewer_experiments/audit_2026-11-22_mac/audit_report.json) records checks performed on the saved results. To reproduce that audit from the repository root, run the following command in the project Python environment with the EMS workbook and EMS code available.

```sh
python -m tools.audit_reviewer_results --output work/reviewer_audit_new
```

The supplementary figure uses the [2/1/1 tuning AUC data](../outputs/reviewer_experiments/audit_2026-11-22_mac/alpha_calibration_211.csv). Compile its source separately from within the `supplement/` directory.

Included source assets:

- main.tex and references.bib
- the three figures referenced by the manuscript
- scripts/generate_revision_figures.py for reproducible regeneration
- the subject-level case-analysis CSV and the LIME explanation for subject 302
- source outputs for both dataset-optimization experiments

Before submission, confirm the author order, affiliations, ORCID and corresponding-author details, the RIVF 2026 template, and the page limit. Recompile in the submission environment and check the final layout.
