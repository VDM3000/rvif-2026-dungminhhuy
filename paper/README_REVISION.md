# Dataset-optimization revision

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

2026_RIVF_teacher_revision_preview_before_dataset_optimization.pdf is an older preview and does not contain the new analyses. The current `main.tex` includes the three-seed and alpha analyses and the separated 2/1/1 late/feature-level fusion results; all added or revised manuscript text is blue. `main.pdf` is the seven-page verification PDF compiled locally with portable Tectonic. The two older tables still trigger overfull-line warnings; the new 2/1/1 table does not. Recompile the source in the final submission environment and check the resulting PDF before submission.

Included source assets:

- main.tex and references.bib
- the three figures referenced by the manuscript
- scripts/generate_revision_figures.py for reproducible regeneration
- the subject-level case-analysis CSV and the LIME explanation for subject 302
- source outputs for both dataset-optimization experiments

Before submission, confirm the author order, affiliations, ORCID/corresponding-author details, the official RIVF 2026 template, and its page limit. The additional section will increase the manuscript beyond the previous seven-page preview; shorten the paper structurally if required.
