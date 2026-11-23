"""Reproduce the fixed P+S/HGB late fusion for three seeds and inspect alpha."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


FOLDS = tuple(f"Set_{i}" for i in range(4))
SEEDS = (42, 17, 2026)
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    binned = np.clip(scores * 255.0, 0, 255).astype(np.uint8)
    raw_threshold, _ = cv2.threshold(binned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(raw_threshold / 255.0)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "acc": accuracy_score(labels, predicted),
        "sensitivity": recall_score(labels, predicted, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "auc": roc_auc_score(labels, scores),
        "precision": precision_score(labels, predicted, zero_division=0),
        "f1_score": f1_score(labels, predicted, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fold_metrics(frame: pd.DataFrame, score_column: str, folds: tuple[str, ...] = FOLDS) -> pd.DataFrame:
    rows = [{"set_name": fold, **metrics(part.label, part[score_column])}
            for fold in folds if not (part := frame.loc[frame.set_name == fold]).empty]
    result = pd.DataFrame(rows)
    average = {"set_name": "average"}
    for column in ("threshold", "acc", "sensitivity", "specificity", "auc", "precision", "f1_score"):
        average[column] = result[column].mean()
    for column in ("tn", "fp", "fn", "tp"):
        average[column] = np.nan
    return pd.concat((result, pd.DataFrame([average])), ignore_index=True)


def check_subjects(features: pd.DataFrame, msnet: pd.DataFrame, columns: list[str]) -> None:
    if len(columns) != 173 or len(set(columns)) != 173:
        raise ValueError("Pupil+Scanpath must contain 173 unique columns")
    if not set(columns).issubset(features.columns):
        raise ValueError("Behavioral feature columns are missing")
    for name, frame in (("features", features), ("msnet", msnet)):
        counts = frame.groupby("set_name").size().to_dict()
        if len(frame) != 160 or frame.subject_int.nunique() != 160 or counts != {f: 40 for f in FOLDS}:
            raise ValueError(f"{name} must have 160 unique subjects in four 40-person folds: {counts}")
    left = features[["subject_int", "set_name", "label"]].sort_values("subject_int").reset_index(drop=True)
    right = msnet[["subject_int", "set_name", "label"]].sort_values("subject_int").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Subject/fold/label mismatch between behavioral and MSNet inputs")


def run_seed(features: pd.DataFrame, msnet: pd.DataFrame, columns: list[str], seed: int):
    """Return subject predictions, fold metrics, and selected alpha per held-out fold."""
    check_subjects(features, msnet, columns)
    predictions = []
    for fold in FOLDS:
        train = features.loc[features.set_name != fold]
        validation = features.loc[features.set_name == fold]
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03,
                 max_leaf_nodes=15, l2_regularization=0.01, random_state=seed)),
        ])
        model.fit(train[columns], train.label.astype(int))
        part = validation[["subject_int", "set_name", "label"]].copy()
        part["behavior_prob"] = model.predict_proba(validation[columns])[:, 1]
        predictions.append(part)
    behavior = pd.concat(predictions, ignore_index=True)
    merged = msnet[["subject_int", "set_name", "label", "msnet_prob"]].merge(
        behavior[["subject_int", "set_name", "behavior_prob"]],
        on=["subject_int", "set_name"], how="inner", validate="one_to_one",
    )
    if len(merged) != 160:
        raise ValueError("Behavioral/MSNet merge lost subjects")
    chosen = []
    evaluated = []
    for fold in FOLDS:
        tuning = merged.loc[merged.set_name != fold]
        candidates = []
        for alpha in ALPHAS:
            trial = tuning.assign(fused_prob=alpha * tuning.msnet_prob + (1 - alpha) * tuning.behavior_prob)
            avg = fold_metrics(trial, "fused_prob", tuple(f for f in FOLDS if f != fold)).iloc[-1]
            candidates.append({"alpha_msnet": alpha, "auc": avg.auc, "f1_score": avg.f1_score, "acc": avg.acc})
        ranked = pd.DataFrame(candidates).sort_values(["auc", "f1_score", "acc"], ascending=False).reset_index(drop=True)
        best = ranked.iloc[0]
        alpha = float(best.alpha_msnet)
        part = merged.loc[merged.set_name == fold].copy()
        part["alpha_msnet"] = alpha
        part["behavior_weight"] = 1 - alpha
        part["fused_prob"] = alpha * part.msnet_prob + (1 - alpha) * part.behavior_prob
        evaluated.append(part)
        chosen.append({"val_set": fold, "chosen_alpha_msnet": alpha,
                       "chosen_behavior_weight": 1 - alpha, "alpha_search_best_auc": float(best.auc),
                       "alpha_search_best_f1": float(best.f1_score), "alpha_search_best_acc": float(best.acc)})
    result = pd.concat(evaluated, ignore_index=True).sort_values(["set_name", "subject_int"]).reset_index(drop=True)
    result.insert(1, "subject_id", result.subject_int.map(lambda sid: f"{int(sid):03d}"))
    return result, fold_metrics(result, "fused_prob"), pd.DataFrame(chosen)


def alpha_curve(predictions: pd.DataFrame, alphas: np.ndarray = ALPHAS) -> pd.DataFrame:
    """AUC at each alpha and fold, with a within-fold MSNet reference."""
    baseline = {fold: roc_auc_score(part.label, part.msnet_prob)
                for fold in FOLDS for part in [predictions.loc[predictions.set_name == fold]]}
    rows = []
    for alpha in alphas:
        for fold in FOLDS:
            part = predictions.loc[predictions.set_name == fold]
            auc = roc_auc_score(part.label, alpha * part.msnet_prob + (1 - alpha) * part.behavior_prob)
            delta = auc - baseline[fold]
            if abs(delta) < 1e-12:
                delta = 0.0
            rows.append({"alpha_msnet": float(alpha), "behavior_weight": 1 - float(alpha),
                         "set_name": fold, "auc": auc, "msnet_auc": baseline[fold],
                         "delta_auc_vs_msnet": delta})
    return pd.DataFrame(rows)


def summarize_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    values = seed_metrics.drop(columns="seed")
    rows = {}
    for column in values:
        series = values[column].astype(float)
        rows[column] = {"mean": float(series.mean()),
                        "sample_variance": 0.0 if series.nunique(dropna=False) == 1 else float(series.var(ddof=1))}
    return pd.DataFrame.from_dict(rows, orient="index")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline
    if (baseline / "improve_behavioral").is_dir():
        baseline = baseline / "improve_behavioral"
    features = pd.read_csv(baseline / "subject_behavior_features.csv")
    msnet = pd.read_csv(baseline / "msnet_prob_val.csv")
    columns = json.loads((baseline / "feature_sets_clean.json").read_text(encoding="utf-8"))["pupil_scanpath"]
    baseline_manifest = args.baseline / "manifest.json"
    source_hashes = {name: sha256(baseline / name) for name in
                     ("subject_behavior_features.csv", "msnet_prob_val.csv", "feature_sets_clean.json")}
    if baseline_manifest.is_file():
        source_hashes["baseline_manifest.json"] = sha256(baseline_manifest)
    source_hashes["seed_alpha.py"] = sha256(Path(__file__))
    repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    fold_rows = []
    seed42_predictions = None
    for seed in SEEDS:
        folder = args.output / f"seed_{seed}"
        folder.mkdir()
        predictions, foldwise, alpha = run_seed(features, msnet, columns, seed)
        predictions.to_csv(folder / "predictions.csv", index=False)
        foldwise.to_csv(folder / "fold_metrics.csv", index=False)
        alpha.to_csv(folder / "alpha_by_fold.csv", index=False)
        (folder / "manifest.json").write_text(json.dumps({
            "seed": seed, "feature_set": "pupil_scanpath", "feature_count": len(columns),
            "model": "HistGradientBoostingClassifier", "msnet": "fixed pretrained OOF probabilities",
            "alpha_grid": [float(x) for x in ALPHAS], "sklearn": sklearn.__version__,
            "source_sha256": source_hashes, "repo_commit": repo_commit,
            "selection_rule": "other three OOF folds; mean AUC, F1 then accuracy",
            "limitations": "historical four-fold selection, not independent test or confidence interval",
            "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        }, indent=2), encoding="utf-8")
        rows.append({"seed": seed, **foldwise.iloc[-1][["acc", "sensitivity", "specificity", "auc", "precision", "f1_score"]].to_dict()})
        fold_rows.extend({"seed": seed, **row} for row in foldwise.iloc[:4].to_dict("records"))
        if seed == 42:
            seed42_predictions = predictions
        print(f"seed={seed} auc={foldwise.iloc[-1].auc:.6f}", flush=True)
    seed_metrics = pd.DataFrame(rows)
    seed_metrics.to_csv(args.output / "seed_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output / "seed_fold_metrics.csv", index=False)
    summary = summarize_seed_metrics(seed_metrics)
    summary.to_csv(args.output / "seed_summary.csv")
    historical = pd.read_csv(baseline / "best_fusion_prob_val.csv").sort_values("subject_int")
    reproduced = seed42_predictions.sort_values("subject_int")
    if not np.allclose(historical.behavior_prob, reproduced.behavior_prob, rtol=0, atol=1e-12):
        raise RuntimeError("Seed 42 behavioral predictions do not match the frozen baseline")
    if not np.allclose(historical.fused_prob, reproduced.fused_prob, rtol=0, atol=1e-12):
        raise RuntimeError("Seed 42 fused predictions do not match the frozen baseline")
    curve = alpha_curve(seed42_predictions)
    curve.to_csv(args.output / "alpha_curve_by_fold.csv", index=False)
    mean_curve = curve.groupby("alpha_msnet", as_index=False)[["auc", "delta_auc_vs_msnet"]].mean()
    mean_curve.to_csv(args.output / "alpha_curve_mean.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean_curve.alpha_msnet, mean_curve.delta_auc_vs_msnet * 100, marker="o", markersize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="MSNet fusion weight alpha", ylabel="Mean AUC change vs MSNet (points)")
    fig.tight_layout()
    fig.savefig(args.output / "alpha_delta_auc.png", dpi=200)
    plt.close(fig)
    negative_count = int((curve.delta_auc_vs_msnet < -1e-12).sum())
    negative_by_fold = curve.groupby("set_name")["delta_auc_vs_msnet"].apply(lambda values: int((values < -1e-12).sum()))
    identical_predictions = all((args.output / "seed_42" / "predictions.csv").read_bytes() ==
                                (args.output / f"seed_{seed}" / "predictions.csv").read_bytes() for seed in SEEDS)
    (args.output / "interpretation.txt").write_text(
        f"Historical validation only; not an external test. All three seeds have identical subject predictions: {identical_predictions}. "
        f"Negative fold-level AUC deltas: {negative_count}/{len(curve)}; by fold: {negative_by_fold.to_dict()}. "
        f"Negative mean deltas: {int((mean_curve.delta_auc_vs_msnet < -1e-12).sum())}/{len(mean_curve)}. "
        "AUC is nonnegative by definition; three seeds do not provide a 95% confidence interval.\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
