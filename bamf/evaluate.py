"""Audit 160 independent 2/1/1 evaluations and compare late/early fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import FOLDS, fold_roles, load_subject_map, otsu_threshold
from .train_msnet import fixed_threshold_metrics


METRICS = ("acc", "sensitivity", "specificity", "auc", "precision", "f1_score")


def audit_evaluations(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    if actual.subject_int.duplicated().any():
        raise ValueError("duplicate evaluated subject IDs")
    columns = ["subject_int", "set_name", "label"]
    left = actual[columns].sort_values("subject_int").reset_index(drop=True)
    right = expected[columns].sort_values("subject_int").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Evaluated subject, fold, or label mismatch")


def audit_prediction_metrics(part: pd.DataFrame, detail: dict, score_column: str) -> dict:
    """Recompute saved metrics and reject stale scores, thresholds, or labels."""
    scores = part[score_column].to_numpy(dtype=float)
    thresholds = part.threshold.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Invalid prediction scores")
    if not np.isfinite(thresholds).all() or not np.allclose(thresholds, detail["threshold"], rtol=0, atol=1e-12):
        raise ValueError("Prediction threshold mismatch")
    predicted = (scores >= float(detail["threshold"])).astype(int)
    if not np.array_equal(predicted, part.pred.to_numpy()):
        raise ValueError("Predicted label mismatch")
    measured = fixed_threshold_metrics(part.label, scores, float(detail["threshold"]))
    for key in METRICS:
        if not np.isclose(measured[key], detail[key], rtol=0, atol=1e-12):
            raise ValueError(f"Saved metric mismatch: {key}")
    return measured


def collect(root: Path, expected: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, results = [], []
    for fold in FOLDS:
        folder = root / fold
        complete = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
        role_train, role_alpha, role_eval = fold_roles(fold)
        detail = complete["result"]
        if (detail["fold"] != role_eval or detail["alpha_fold"] != role_alpha or
                tuple(detail["train_folds"]) != role_train):
            raise ValueError(f"Wrong 2/1/1 roles in {folder}")
        part = pd.read_csv(folder / "evaluation_predictions.csv")
        if len(part) != 40 or set(part.set_name) != {fold}:
            raise ValueError(f"Expected 40 evaluation predictions in {folder}")
        score_column = "fused_prob" if method == "Late Fusion 2/1/1" else "early_prob"
        measured = audit_prediction_metrics(part, detail, score_column)
        part["method"] = method
        predictions.append(part)
        results.append({"method": method, "fold": fold, **{key: measured[key] for key in METRICS}})
    all_predictions = pd.concat(predictions, ignore_index=True)
    audit_evaluations(all_predictions, expected)
    return all_predictions, pd.DataFrame(results)


def msnet_from_calibration(calibration: pd.DataFrame, evaluated: pd.DataFrame):
    """Use the 40 tuning subjects' scores to set MSNet's decision threshold."""
    threshold = otsu_threshold(calibration.msnet_prob)
    predictions = evaluated[["subject_int", "set_name", "label", "msnet_prob"]].copy()
    if "subject_id" in evaluated:
        predictions.insert(1, "subject_id", evaluated.subject_id)
    predictions["threshold"] = threshold
    predictions["pred"] = (predictions.msnet_prob >= threshold).astype(int)
    predictions["method"] = "MSNet 2/1/1"
    result = fixed_threshold_metrics(predictions.label, predictions.msnet_prob, threshold)
    return predictions, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--late-root", type=Path, required=True)
    parser.add_argument("--early-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expected = load_subject_map(args.workbook)
    late_predictions, late_metrics = collect(args.late_root, expected, "Late Fusion 2/1/1")
    baseline_predictions, baseline_metrics = [], []
    for fold in FOLDS:
        calibration = pd.read_csv(args.late_root / fold / "calibration_predictions.csv")
        evaluated = late_predictions.loc[late_predictions.set_name == fold]
        prediction, metric = msnet_from_calibration(calibration, evaluated)
        baseline_predictions.append(prediction)
        baseline_metrics.append({"method": "MSNet 2/1/1", "fold": fold,
                                 **{key: metric[key] for key in METRICS}})
    baseline_predictions = pd.concat(baseline_predictions, ignore_index=True)
    audit_evaluations(baseline_predictions, expected)
    data = [(baseline_predictions, pd.DataFrame(baseline_metrics)), (late_predictions, late_metrics)]
    if args.early_root is not None:
        early_predictions, early_metrics = collect(args.early_root, expected, "Early Fusion 2/1/1")
        data.append((early_predictions, early_metrics))
    args.output.mkdir(parents=True)
    all_predictions = pd.concat([item[0] for item in data], ignore_index=True)
    all_metrics = pd.concat([item[1] for item in data], ignore_index=True)
    all_predictions.to_csv(args.output / "all_evaluation_predictions.csv", index=False)
    all_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    summary_rows = []
    for method, part in all_metrics.groupby("method", sort=False):
        row = {"method": method, "protocol": "2/1/1, 80 train/40 tune/40 evaluate"}
        for metric in METRICS:
            row[metric] = float(part[metric].mean())
            row[f"{metric}_fold_sd"] = float(part[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)
    historical = pd.DataFrame([
        {"method": "MSNet historical", "protocol": "selected four-fold validation", "auc": 0.8972166982323232, "acc": 0.83125},
        {"method": "BAMF historical", "protocol": "selected four-fold validation", "auc": 0.9502533143939393, "acc": 0.84375},
    ])
    pd.concat([historical, summary], ignore_index=True).to_csv(args.output / "comparison_with_historical.csv", index=False)
    baseline_paired = baseline_predictions[["subject_int", "set_name", "label", "pred", "msnet_prob"]].rename(
        columns={"pred": "msnet_pred"}).merge(
        late_predictions[["subject_int", "set_name", "pred", "fused_prob"]].rename(columns={"pred": "late_pred"}),
        on=["subject_int", "set_name"], how="inner", validate="one_to_one")
    baseline_paired["msnet_correct"] = baseline_paired.msnet_pred == baseline_paired.label
    baseline_paired["late_correct"] = baseline_paired.late_pred == baseline_paired.label
    baseline_paired.to_csv(args.output / "paired_msnet_late.csv", index=False)
    if args.early_root is not None:
        paired = late_predictions[["subject_int", "set_name", "label", "pred", "fused_prob"]].rename(
            columns={"pred": "late_pred", "fused_prob": "late_prob"}).merge(
            early_predictions[["subject_int", "set_name", "pred", "early_prob"]].rename(columns={"pred": "early_pred"}),
            on=["subject_int", "set_name"], how="inner", validate="one_to_one")
        paired["late_correct"] = paired.late_pred == paired.label
        paired["early_correct"] = paired.early_pred == paired.label
        paired.to_csv(args.output / "paired_subject_comparison.csv", index=False)
    (args.output / "interpretation.txt").write_text(
        "The 2/1/1 estimates use 80 training, 40 tuning, and 40 evaluation subjects per fold. "
        "Historical 95.03% BAMF and 89.72% MSNet used a different selected four-fold protocol "
        "and must not be treated as directly comparable independent test estimates. "
        "No labeled EMS hidden test or external validation is available.\n",
        encoding="utf-8",
    )
    print(summary[["method", "auc", "acc"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
