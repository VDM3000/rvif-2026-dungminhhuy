"""Fixed 2/1/1 subject roles and calibration-only fusion rules."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


FOLDS = tuple(f"Set_{i}" for i in range(4))
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def fold_roles(evaluation_fold: str) -> tuple[tuple[str, str], str, str]:
    """Return (train folds, alpha fold, evaluation fold)."""
    if evaluation_fold not in FOLDS:
        raise ValueError(f"Unknown fold: {evaluation_fold}")
    index = FOLDS.index(evaluation_fold)
    alpha_fold = FOLDS[(index + 1) % 4]
    train_folds = tuple(fold for fold in FOLDS if fold not in (evaluation_fold, alpha_fold))
    return train_folds, alpha_fold, evaluation_fold


def validate_subject_map(frame: pd.DataFrame, expected_per_fold: int = 40) -> None:
    if frame.subject_int.duplicated().any():
        raise ValueError("duplicate subject IDs")
    counts = frame.groupby("set_name").size().to_dict()
    if counts != {fold: expected_per_fold for fold in FOLDS}:
        raise ValueError(f"Unexpected fold sizes: {counts}")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Labels must be 0/1")


def load_subject_map(workbook) -> pd.DataFrame:
    raw = pd.read_excel(workbook)
    rows = []
    for fold in FOLDS:
        for value in raw[fold].dropna():
            subject = int(value)
            rows.append({"subject_int": subject, "subject_id": f"{subject:03d}",
                         "set_name": fold, "label": int(subject >= 200)})
    frame = pd.DataFrame(rows).sort_values(["set_name", "subject_int"]).reset_index(drop=True)
    validate_subject_map(frame)
    return frame


def otsu_threshold(scores) -> float:
    values = np.clip(np.asarray(scores, dtype=float) * 255.0, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold / 255.0)


def select_alpha(labels, msnet_scores, behavior_scores, alphas=ALPHAS) -> dict[str, float]:
    """Choose alpha and one threshold using only calibration-subject arrays."""
    labels = np.asarray(labels, dtype=int)
    msnet_scores = np.asarray(msnet_scores, dtype=float)
    behavior_scores = np.asarray(behavior_scores, dtype=float)
    if not (len(labels) == len(msnet_scores) == len(behavior_scores)) or len(labels) == 0:
        raise ValueError("Calibration arrays must be nonempty and aligned")
    candidates = []
    for alpha in alphas:
        score = float(alpha) * msnet_scores + (1 - float(alpha)) * behavior_scores
        threshold = otsu_threshold(score)
        predicted = (score >= threshold).astype(int)
        candidates.append({"alpha_msnet": float(alpha), "threshold": threshold,
                           "auc": float(roc_auc_score(labels, score)),
                           "f1_score": float(f1_score(labels, predicted, zero_division=0)),
                           "acc": float(accuracy_score(labels, predicted))})
    ranked = pd.DataFrame(candidates).sort_values(["auc", "f1_score", "acc", "alpha_msnet"],
                                                  ascending=[False, False, False, True])
    return {key: float(value) for key, value in ranked.iloc[0].to_dict().items()}
