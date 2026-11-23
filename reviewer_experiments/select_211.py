"""Select one of 21 behavioral configurations inside the 80 training subjects of 2/1/1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .protocol_211 import FOLDS, fold_roles, otsu_threshold, select_alpha, validate_subject_map
from .seed_alpha import sha256
from .train_211 import fixed_threshold_metrics


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/reviewer_experiments"


def make_behavior_model(name: str, seed: int) -> Pipeline:
    """Match the three classifiers and fixed hyperparameters in windows.ipynb."""
    if name == "logreg":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                    solver="liblinear", random_state=seed))])
    if name == "rf":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(n_estimators=500, max_depth=None,
                                                        min_samples_leaf=2, class_weight="balanced",
                                                        random_state=seed, n_jobs=-1))])
    if name == "hgb":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03,
                                                                 max_leaf_nodes=15, l2_regularization=0.01,
                                                                 random_state=seed))])
    raise ValueError(name)


def choose_configuration(candidates: pd.DataFrame) -> pd.Series:
    """Rank inner-fold mean AUC, F1, accuracy; lexical key resolves exact ties."""
    required = {"feature_set", "model", "auc", "f1_score", "acc"}
    if not required.issubset(candidates.columns) or len(candidates) != 21:
        raise ValueError("Expected 21 scored configurations")
    if candidates[["feature_set", "model"]].duplicated().any():
        raise ValueError("Duplicate configuration")
    return candidates.sort_values(["auc", "f1_score", "acc", "feature_set", "model"],
                                  ascending=[False, False, False, True, True]).iloc[0]


def select_inside_training(features: pd.DataFrame, feature_sets: dict[str, list[str]],
                           train_folds: tuple[str, str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Two inner 40/40 rotations. Neither alpha nor evaluation fold enters selection."""
    if len(feature_sets) != 7 or set(features.set_name.unique()) != set(FOLDS):
        raise ValueError("Expected seven feature sets and four folds")
    rows = []
    for feature_set, columns in feature_sets.items():
        for name in ("logreg", "rf", "hgb"):
            for inner_val_fold in train_folds:
                inner_train_fold = next(f for f in train_folds if f != inner_val_fold)
                train = features.loc[features.set_name == inner_train_fold]
                validation = features.loc[features.set_name == inner_val_fold]
                if (len(train), len(validation)) != (40, 40):
                    raise ValueError("Inner split must contain 40 train and 40 validation subjects")
                model = make_behavior_model(name, seed)
                model.fit(train[columns], train.label.astype(int))
                scores = model.predict_proba(validation[columns])[:, 1]
                threshold = otsu_threshold(scores)
                rows.append({"feature_set": feature_set, "model": name,
                             "inner_train_fold": inner_train_fold, "inner_val_fold": inner_val_fold,
                             "auc": roc_auc_score(validation.label, scores),
                             "f1_score": f1_score(validation.label, scores >= threshold, zero_division=0),
                             "acc": accuracy_score(validation.label, scores >= threshold)})
    inner = pd.DataFrame(rows)
    ranked = inner.groupby(["feature_set", "model"], as_index=False)[["auc", "f1_score", "acc"]].mean()
    winner = choose_configuration(ranked)
    return inner, ranked.sort_values(["auc", "f1_score", "acc", "feature_set", "model"],
                                      ascending=[False, False, False, True, True]), winner


def load_msnet_part(folder: Path, filename: str, expected: pd.DataFrame) -> pd.DataFrame:
    part = pd.read_csv(folder / filename)
    check = expected[["subject_int", "set_name", "label"]].merge(
        part[["subject_int", "set_name", "label", "msnet_prob"]],
        on=["subject_int", "set_name", "label"], how="left", validate="one_to_one")
    if len(check) != 40 or check.msnet_prob.isna().any() or len(part) != 40:
        raise ValueError(f"MSNet predictions do not match expected subjects: {filename}")
    return check


def run_fold(features: pd.DataFrame, feature_sets: dict[str, list[str]], late_root: Path,
             output_root: Path, evaluation_fold: str, seed: int) -> None:
    train_folds, alpha_fold, eval_fold = fold_roles(evaluation_fold)
    folder = output_root / eval_fold
    if folder.exists():
        raise FileExistsError(f"Choose a fresh output directory: {folder}")
    folder.mkdir(parents=True)
    base = late_root / eval_fold
    base_config = json.loads((base / "run_config.json").read_text(encoding="utf-8"))
    if (tuple(base_config["train_folds"]), base_config["alpha_fold"], base_config["eval_fold"]) != (
            train_folds, alpha_fold, eval_fold):
        raise ValueError("Existing MSNet run used different 2/1/1 roles")
    manifest = {"protocol": "2/1/1 with 21-config inner selection on 80 train subjects",
                "train_folds": train_folds, "alpha_fold": alpha_fold, "evaluation_fold": eval_fold,
                "seed": seed, "ranking": "inner mean AUC, F1, accuracy, lexical key",
                "alpha_grid": "0:0.05:1, selected only on alpha fold",
                "source_sha256": sha256(Path(__file__)),
                "features_sha256": sha256(ROOT / "outputs/improve_behavioral/subject_behavior_features.csv"),
                "feature_sets_sha256": sha256(ROOT / "outputs/improve_behavioral/feature_sets_clean.json"),
                "msnet_checkpoint_sha256": sha256(base / "final_model.pt")}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    inner, ranked, winner = select_inside_training(features, feature_sets, train_folds, seed)
    inner.to_csv(folder / "inner_fold_metrics.csv", index=False)
    ranked.to_csv(folder / "configuration_ranking.csv", index=False)
    columns = feature_sets[str(winner.feature_set)]
    model = make_behavior_model(str(winner.model), seed)
    training = features.loc[features.set_name.isin(train_folds)]
    model.fit(training[columns], training.label.astype(int))
    parts = []
    for fold, filename in ((alpha_fold, "calibration_predictions.csv"),
                           (eval_fold, "evaluation_predictions.csv")):
        held = features.loc[features.set_name == fold]
        msnet = load_msnet_part(base, filename, held)
        msnet["behavior_prob"] = model.predict_proba(held[columns])[:, 1]
        parts.append(msnet)
    calibration, evaluated = parts
    choice = select_alpha(calibration.label, calibration.msnet_prob, calibration.behavior_prob)
    for part in parts:
        part["alpha_msnet"] = choice["alpha_msnet"]
        part["fused_prob"] = choice["alpha_msnet"] * part.msnet_prob + (1 - choice["alpha_msnet"]) * part.behavior_prob
        part["threshold"] = choice["threshold"]
        part["pred"] = (part.fused_prob >= choice["threshold"]).astype(int)
    calibration.to_csv(folder / "calibration_predictions.csv", index=False)
    evaluated.to_csv(folder / "evaluation_predictions.csv", index=False)
    result = {"evaluation_fold": eval_fold, "train_folds": str(train_folds), "alpha_fold": alpha_fold,
              "feature_set": str(winner.feature_set), "model": str(winner.model),
              "inner_auc": float(winner.auc), "alpha_msnet": choice["alpha_msnet"],
              **fixed_threshold_metrics(evaluated.label, evaluated.fused_prob, choice["threshold"])}
    pd.DataFrame([result]).to_csv(folder / "evaluation_metrics.csv", index=False)
    print(f"{eval_fold}: {winner.feature_set}/{winner.model}, alpha={choice['alpha_msnet']:.2f}, "
          f"eval AUC={result['auc']:.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--late-root", type=Path, default=RUNS / "late_211_2026-11-22")
    parser.add_argument("--output-root", type=Path, default=RUNS / "selected_211_final_2026-11-23")
    parser.add_argument("--eval-fold", choices=[*FOLDS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    features = pd.read_csv(ROOT / "outputs/improve_behavioral/subject_behavior_features.csv")
    feature_sets = json.loads((ROOT / "outputs/improve_behavioral/feature_sets_clean.json").read_text(encoding="utf-8"))
    validate_subject_map(features)
    for fold in (FOLDS if args.eval_fold == "all" else (args.eval_fold,)):
        run_fold(features, feature_sets, args.late_root, args.output_root, fold, args.seed)
    if all((args.output_root / fold / "evaluation_predictions.csv").exists() for fold in FOLDS):
        rows = [pd.read_csv(args.output_root / fold / "evaluation_metrics.csv") for fold in FOLDS]
        pd.concat(rows, ignore_index=True).to_csv(args.output_root / "fold_metrics.csv", index=False)
        predictions = pd.concat([pd.read_csv(args.output_root / fold / "evaluation_predictions.csv")
                                 for fold in FOLDS], ignore_index=True)
        if len(predictions) != 160 or predictions.subject_int.nunique() != 160:
            raise ValueError("Expected 160 unique evaluation subjects")
        predictions.to_csv(args.output_root / "all_evaluation_predictions.csv", index=False)
        summary = pd.concat(rows, ignore_index=True)[["acc", "sensitivity", "specificity", "auc", "precision", "f1_score"]].mean()
        summary.to_frame("mean").to_csv(args.output_root / "summary_metrics.csv")


if __name__ == "__main__":
    main()
