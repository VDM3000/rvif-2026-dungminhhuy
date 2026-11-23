"""Evaluate Late Fusion using held-out tuning and evaluation folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from .splits import FOLDS, alpha_candidates, fold_roles, validate_subject_map
from .seed_analysis import sha256
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from .train_msnet import fixed_threshold_metrics


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("acc", "sensitivity", "specificity", "auc", "precision", "f1_score")


def rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Selection uses middle-fold scores only; numerical and lexical ties are stable."""
    expected = {"feature_set", "model", "alpha_msnet", "threshold", "auc", "f1_score", "acc"}
    if not expected.issubset(candidates.columns) or candidates.empty:
        raise ValueError("Missing candidate columns")
    if candidates[["feature_set", "model", "alpha_msnet"]].duplicated().any():
        raise ValueError("Duplicate feature/model/alpha candidate")
    # On 40 people, mathematically equal metrics may differ below 1e-12 in
    # floating arithmetic. Such noise must not override the F1/accuracy rule.
    ranked = candidates.assign(_auc=candidates.auc.round(12),
                               _f1=candidates.f1_score.round(12),
                               _acc=candidates.acc.round(12)).sort_values(
        ["_auc", "_f1", "_acc", "feature_set", "model", "alpha_msnet"],
        ascending=[False, False, False, True, True, True], kind="mergesort",
    ).drop(columns=["_auc", "_f1", "_acc"]).reset_index(drop=True)
    return ranked


def aligned_msnet(root: Path, fold: str, filename: str, expected: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(root / fold / filename)
    merged = expected[["subject_int", "subject_id", "set_name", "label"]].merge(
        source[["subject_int", "msnet_prob"]], on="subject_int", how="left", validate="one_to_one"
    )
    if len(source) != 40 or len(merged) != 40 or merged.msnet_prob.isna().any():
        raise ValueError(f"MSNet predictions missing or duplicated for {fold}/{filename}")
    check = expected[["subject_int", "set_name", "label"]].merge(
        source[["subject_int", "set_name", "label"]], on=["subject_int", "set_name", "label"],
        validate="one_to_one",
    )
    if len(check) != 40:
        raise ValueError("MSNet fold or labels differ from behavioral data")
    return merged


def one_fold(args, features: pd.DataFrame, feature_sets: dict[str, list[str]],
             eval_fold: str) -> None:
    train_folds, selection_fold, _ = fold_roles(eval_fold)
    destination = args.output_root / eval_fold
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; choose a new --output-root")
    msnet_folder = args.msnet_root / eval_fold
    config = json.loads((msnet_folder / "run_config.json").read_text(encoding="utf-8"))
    if (tuple(config["train_folds"]), config["alpha_fold"], config["eval_fold"]) != (
            train_folds, selection_fold, eval_fold):
        raise ValueError("MSNet checkpoint was trained with different roles")
    completion = json.loads((msnet_folder / "completed.json").read_text(encoding="utf-8"))
    if sha256(msnet_folder / "final_model.pt") != completion["checkpoint_sha256"]:
        raise ValueError("MSNet checkpoint hash differs from its completed record")
    training = features.loc[features.set_name.isin(train_folds)]
    selection = features.loc[features.set_name == selection_fold]
    evaluated = features.loc[features.set_name == eval_fold]
    if (len(training), len(selection), len(evaluated)) != (80, 40, 40):
        raise ValueError("Expected 80/40/40 subjects")
    destination.mkdir(parents=True)
    manifest = {
        "created_before_run": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "eval_fold": eval_fold, "training_folds": train_folds,
        "selection_fold": selection_fold, "seed": args.seed,
        "rule": "Fit pupil and scanpath HGB on training folds; tune alpha on the tuning fold.",
        "tie_break": "AUC, F1, accuracy rounded to 12 decimals descending; feature, model, alpha ascending",
        "source_sha256": sha256(Path(__file__)),
        "behavior_sha256": sha256(args.behavior_features),
        "feature_sets_sha256": sha256(args.feature_sets),
        "msnet_checkpoint_sha256": sha256(msnet_folder / "final_model.pt"),
        "msnet_config_sha256": sha256(msnet_folder / "run_config.json"),
        "python": sys.version.split()[0],
        "repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True, check=True).stdout.strip(),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    msnet_selection = aligned_msnet(args.msnet_root, eval_fold, "calibration_predictions.csv", selection)
    columns = feature_sets["pupil_scanpath"]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                       l2_regularization=0.01, random_state=args.seed),
    )
    model.fit(training[columns], training.label.astype(int))
    scores = model.predict_proba(selection[columns])[:, 1]
    options = alpha_candidates(selection.label, msnet_selection.msnet_prob, scores)
    options.insert(0, "model", "hgb")
    options.insert(0, "feature_set", "pupil_scanpath")
    options.insert(2, "n_features", len(columns))
    candidates = rank_candidates(options)
    candidates.to_csv(destination / "selection_candidates.csv", index=False)
    winner = candidates.iloc[0]
    chosen_set, chosen_model = "pupil_scanpath", "hgb"
    joblib.dump(model, destination / "behavior_model.joblib")
    msnet_eval = aligned_msnet(args.msnet_root, eval_fold, "evaluation_predictions.csv", evaluated)
    for part, source in ((msnet_selection, selection), (msnet_eval, evaluated)):
        behavior_scores = model.predict_proba(source[columns])[:, 1]
        part["behavior_prob"] = behavior_scores
        part["alpha_msnet"] = float(winner.alpha_msnet)
        part["fused_prob"] = float(winner.alpha_msnet) * part.msnet_prob + (1 - float(winner.alpha_msnet)) * behavior_scores
        part["threshold"] = float(winner.threshold)
        part["pred"] = (part.fused_prob >= float(winner.threshold)).astype(int)
        part["feature_set"] = chosen_set
        part["model"] = chosen_model
    msnet_selection.to_csv(destination / "selection_predictions.csv", index=False)
    msnet_eval.to_csv(destination / "evaluation_predictions.csv", index=False)
    result = {"eval_fold": eval_fold, "selection_fold": selection_fold,
              "training_folds": str(train_folds), "feature_set": chosen_set, "model": chosen_model,
              "alpha_msnet": float(winner.alpha_msnet), "selection_auc": float(winner.auc),
              **fixed_threshold_metrics(msnet_eval.label, msnet_eval.fused_prob, float(winner.threshold)),
              "msnet_auc": fixed_threshold_metrics(msnet_eval.label, msnet_eval.msnet_prob,
                                                   float(winner.threshold))["auc"]}
    pd.DataFrame([result]).to_csv(destination / "evaluation_metrics.csv", index=False)
    (destination / "completed.json").write_text(json.dumps({
        "manifest_sha256": sha256(destination / "manifest.json"),
        "behavior_model_sha256": sha256(destination / "behavior_model.joblib"),
        "selection_candidates_sha256": sha256(destination / "selection_candidates.csv"),
    }, indent=2), encoding="utf-8")
    print(f"{eval_fold}: {chosen_set}/{chosen_model}, alpha={winner.alpha_msnet:.2f}, "
          f"AUC={result['auc']:.6f}", flush=True)


def summarize(root: Path) -> None:
    rows = pd.concat([pd.read_csv(root / fold / "evaluation_metrics.csv") for fold in FOLDS], ignore_index=True)
    predictions = pd.concat([pd.read_csv(root / fold / "evaluation_predictions.csv") for fold in FOLDS], ignore_index=True)
    if len(rows) != 4 or len(predictions) != 160 or predictions.subject_int.nunique() != 160:
        raise ValueError("Expected four metrics and 160 unique evaluation subjects")
    for fold in FOLDS:
        folder = root / fold
        saved = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
        for name, file in (("manifest_sha256", "manifest.json"),
                           ("behavior_model_sha256", "behavior_model.joblib"),
                           ("selection_candidates_sha256", "selection_candidates.csv")):
            if saved[name] != sha256(folder / file):
                raise ValueError(f"Saved hash mismatch: {folder / file}")
        selected = pd.read_csv(folder / "selection_candidates.csv").iloc[0]
        reported = rows.loc[rows.eval_fold == fold].iloc[0]
        if (selected.feature_set != reported.feature_set or selected.model != reported.model or
                not np.isclose(selected.alpha_msnet, reported.alpha_msnet)):
            raise ValueError(f"Selected configuration differs from reported result: {fold}")
        part = predictions.loc[predictions.set_name == fold]
        if len(part) != 40 or not np.isclose(
                fixed_threshold_metrics(part.label, part.fused_prob, float(reported.threshold))["auc"],
                reported.auc):
            raise ValueError(f"Stale evaluation metrics: {fold}")
    folder = root
    rows.to_csv(folder / "fold_metrics.csv", index=False)
    predictions.to_csv(folder / "all_evaluation_predictions.csv", index=False)
    rows[list(METRICS) + ["msnet_auc"]].mean().to_frame("mean").to_csv(folder / "summary_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = ROOT / "outputs/improve_behavioral"
    parser.add_argument("--msnet-root", type=Path, required=True,
                        help="Directory containing the four trained MSNet folds")
    parser.add_argument("--behavior-features", type=Path, default=baseline / "subject_behavior_features.csv")
    parser.add_argument("--feature-sets", type=Path, default=baseline / "feature_sets_clean.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-fold", choices=[*FOLDS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    features = pd.read_csv(args.behavior_features)
    validate_subject_map(features)
    feature_sets = json.loads(args.feature_sets.read_text(encoding="utf-8"))
    columns = feature_sets.get("pupil_scanpath", [])
    if len(columns) != 173 or not set(columns).issubset(features.columns):
        raise ValueError("Expected 173 pupil and scanpath features")
    folds = FOLDS if args.eval_fold == "all" else (args.eval_fold,)
    for fold in folds:
        one_fold(args, features, feature_sets, fold)
    if all((args.output_root / fold / "completed.json").exists() for fold in FOLDS):
        summarize(args.output_root)


if __name__ == "__main__":
    main()
