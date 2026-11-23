from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    CATEGORY_DIRS,
    CATEGORY_ORDER,
    FOLDS,
    FULL_RESULTS_DIR,
    category_dir,
    compute_foldwise_metrics,
    image_sets_from_disk,
    write_json,
)


META_COLUMNS = ["subject_int", "subject_id", "set_name", "label"]
ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
MODEL_NAMES = ("logreg", "rf", "hgb")


def aggregate_category_features(category: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    source = FULL_RESULTS_DIR / "stim_behavior_features.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing stimulus feature table: {source}. Run the baseline notebook first."
        )
    stimulus = pd.read_csv(source)
    image_stems = stimulus["image"].astype(str).map(lambda value: Path(value).stem)
    category_images = image_sets_from_disk()[category]
    stimulus = stimulus[image_stems.isin(category_images)].copy()
    if stimulus["image"].map(lambda value: Path(str(value)).stem).nunique() != len(category_images):
        raise RuntimeError(f"Behavior table does not cover every {category} image")

    stimulus_meta = ["subject_int", "subject_id", "set_name", "label", "image"]
    numeric_columns = [column for column in stimulus.columns if column not in stimulus_meta]
    aggregate_functions = ["mean", "std", "median", "min", "max"]
    subject_aggregate = stimulus.groupby("subject_int")[numeric_columns].agg(aggregate_functions)
    subject_aggregate.columns = [
        f"{column}_{function}" for column, function in subject_aggregate.columns
    ]
    subject_aggregate = subject_aggregate.reset_index()
    subject_meta = stimulus[META_COLUMNS].drop_duplicates("subject_int")
    valid_images = (
        stimulus.groupby("subject_int")["image"].nunique().reset_index(name="num_valid_images")
    )
    total_fixations = (
        stimulus.groupby("subject_int")["fixation_count"]
        .sum()
        .reset_index(name="num_total_fixations")
    )
    mean_fixations = (
        stimulus.groupby("subject_int")["fixation_count"]
        .mean()
        .reset_index(name="mean_fixations_per_image")
    )
    subject = (
        subject_meta.merge(subject_aggregate, on="subject_int", how="left")
        .merge(valid_images, on="subject_int", how="left")
        .merge(total_fixations, on="subject_int", how="left")
        .merge(mean_fixations, on="subject_int", how="left")
        .sort_values("subject_int")
        .reset_index(drop=True)
    )
    if len(subject) != 160:
        raise RuntimeError(f"Expected 160 subjects for {category}, found {len(subject)}")

    all_features = [column for column in subject.columns if column not in META_COLUMNS]
    duration = sorted(column for column in all_features if column.startswith("duration_"))
    pupil = sorted(column for column in all_features if column.startswith("pupil_"))
    scanpath_bases = [
        "fixation_count",
        "x_mean",
        "y_mean",
        "x_std",
        "y_std",
        "x_dispersion",
        "y_dispersion",
        "xy_dispersion_area",
        "distance_to_center_mean",
        "distance_to_center_std",
        "distance_to_center_max",
        "scanpath_length",
        "saccade_amplitude_mean",
        "saccade_amplitude_std",
        "saccade_amplitude_median",
        "saccade_amplitude_max",
        "num_valid_images",
        "num_total_fixations",
        "mean_fixations_per_image",
    ]

    def belongs(column: str) -> bool:
        return any(column == base or column.startswith(base + "_") for base in scanpath_bases)

    scanpath = sorted(column for column in all_features if belongs(column))
    if any(column.startswith(("duration_", "pupil_")) for column in scanpath):
        raise RuntimeError("Scanpath feature set overlaps duration or pupil features")
    feature_sets = {
        "duration_only": duration,
        "pupil_only": pupil,
        "scanpath_only": scanpath,
        "duration_pupil": sorted(set(duration + pupil)),
        "duration_scanpath": sorted(set(duration + scanpath)),
        "pupil_scanpath": sorted(set(pupil + scanpath)),
        "all_behavior": sorted(set(duration + pupil + scanpath)),
    }
    return subject, feature_sets


def make_model(model_name: str):
    if model_name == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_name == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "hgb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.03,
                        max_leaf_nodes=15,
                        l2_regularization=0.01,
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def behavior_cv(subject: pd.DataFrame, columns: list[str], model_name: str) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    for fold in FOLDS:
        train = subject[subject["set_name"] != fold]
        validation = subject[subject["set_name"] == fold].copy()
        model = make_model(model_name)
        model.fit(train[columns], train["label"].astype(int))
        validation["behavior_prob"] = model.predict_proba(validation[columns])[:, 1]
        predictions.append(validation[META_COLUMNS + ["behavior_prob"]])
    return pd.concat(predictions, ignore_index=True).sort_values(["set_name", "subject_int"])


def load_msnet_predictions(category: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in FOLDS:
        path = category_dir(category) / "msnet" / fold / "prob_val.csv"
        if not path.exists():
            raise FileNotFoundError(f"MSNet fold is incomplete: {path}")
        frames.append(pd.read_csv(path))
    result = pd.concat(frames, ignore_index=True).sort_values(["set_name", "subject_int"])
    if len(result) != 160 or result["subject_int"].nunique() != 160:
        raise RuntimeError(f"Invalid MSNet OOF predictions for {category}")
    return result


def choose_alpha(merged: pd.DataFrame, held_out_fold: str) -> tuple[float, pd.DataFrame]:
    selection = merged[merged["set_name"] != held_out_fold]
    rows: list[dict] = []
    for alpha in ALPHA_GRID:
        candidate = selection.copy()
        candidate["fused_prob"] = (
            alpha * candidate["msnet_prob"] + (1 - alpha) * candidate["behavior_prob"]
        )
        average = compute_foldwise_metrics(
            candidate,
            "fused_prob",
            tuple(fold for fold in FOLDS if fold != held_out_fold),
        ).iloc[-1]
        rows.append(
            {
                "alpha_msnet": alpha,
                "auc": average["auc"],
                "f1_score": average["f1_score"],
                "acc": average["acc"],
                "sensitivity": average["sensitivity"],
                "specificity": average["specificity"],
                "precision": average["precision"],
            }
        )
    search = pd.DataFrame(rows).sort_values(
        ["auc", "f1_score", "acc"], ascending=False
    ).reset_index(drop=True)
    return float(search.iloc[0]["alpha_msnet"]), search


def nested_fusion(behavior: pd.DataFrame, msnet: pd.DataFrame):
    merged = msnet.merge(
        behavior[["subject_int", "set_name", "behavior_prob"]],
        on=["subject_int", "set_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 160:
        raise RuntimeError("Behavior/MSNet merge did not retain 160 subjects")
    prediction_frames: list[pd.DataFrame] = []
    alpha_rows: list[dict] = []
    for fold in FOLDS:
        alpha, search = choose_alpha(merged, fold)
        validation = merged[merged["set_name"] == fold].copy()
        validation["alpha_msnet"] = alpha
        validation["behavior_weight"] = 1 - alpha
        validation["fused_prob"] = (
            alpha * validation["msnet_prob"] + (1 - alpha) * validation["behavior_prob"]
        )
        prediction_frames.append(validation)
        alpha_rows.append(
            {
                "val_set": fold,
                "chosen_alpha_msnet": alpha,
                "chosen_behavior_weight": 1 - alpha,
                "alpha_search_best_auc": search.iloc[0]["auc"],
                "alpha_search_best_f1": search.iloc[0]["f1_score"],
                "alpha_search_best_acc": search.iloc[0]["acc"],
            }
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return predictions, compute_foldwise_metrics(predictions, "fused_prob"), pd.DataFrame(alpha_rows)


def run_one(category: str) -> None:
    output = category_dir(category)
    behavior_dir = output / "behavior"
    fusion_dir = output / "fusion"
    behavior_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)
    subject, feature_sets = aggregate_category_features(category)
    subject.to_csv(behavior_dir / "subject_behavior_features.csv", index=False)
    write_json(behavior_dir / "feature_sets_clean.json", feature_sets)

    msnet = load_msnet_predictions(category)
    msnet.to_csv(output / "msnet_prob_val.csv", index=False)
    msnet_metrics = compute_foldwise_metrics(msnet, "msnet_prob")
    msnet_metrics.to_csv(output / "baseline_msnet_foldwise_metrics.csv", index=False)

    behavior_predictions: dict[str, pd.DataFrame] = {}
    behavior_metrics_rows: list[dict] = []
    all_behavior_frames: list[pd.DataFrame] = []
    for feature_name, columns in feature_sets.items():
        for model_name in MODEL_NAMES:
            key = f"{feature_name}__{model_name}"
            print(f"[{category}] behavior {key} ({len(columns)} features)")
            predictions = behavior_cv(subject, columns, model_name)
            metrics = compute_foldwise_metrics(predictions, "behavior_prob")
            average = metrics.iloc[-1]
            behavior_predictions[key] = predictions
            tagged = predictions.copy()
            tagged.insert(0, "key", key)
            all_behavior_frames.append(tagged)
            behavior_metrics_rows.append(
                {
                    "key": key,
                    "feature_set": feature_name,
                    "model": model_name,
                    "n_features": len(columns),
                    **{name: average[name] for name in ["acc", "sensitivity", "specificity", "auc", "precision", "f1_score"]},
                    "threshold_mean": average["threshold"],
                }
            )
    behavior_metrics = pd.DataFrame(behavior_metrics_rows).sort_values(
        ["auc", "f1_score", "acc"], ascending=False
    ).reset_index(drop=True)
    behavior_metrics.to_csv(behavior_dir / "behavior_only_metrics.csv", index=False)
    pd.concat(all_behavior_frames, ignore_index=True).to_csv(
        behavior_dir / "all_behavior_prob_val.csv", index=False
    )
    best_behavior_key = str(behavior_metrics.iloc[0]["key"])
    best_behavior = behavior_predictions[best_behavior_key]
    best_behavior_metrics = compute_foldwise_metrics(best_behavior, "behavior_prob")
    best_behavior.to_csv(behavior_dir / "best_behavior_prob_val.csv", index=False)
    best_behavior_metrics.to_csv(behavior_dir / "best_behavior_foldwise_metrics.csv", index=False)

    fusion_metrics_rows: list[dict] = []
    fusion_predictions: dict[str, pd.DataFrame] = {}
    fusion_fold_metrics: dict[str, pd.DataFrame] = {}
    fusion_alphas: dict[str, pd.DataFrame] = {}
    for key, behavior in behavior_predictions.items():
        print(f"[{category}] fusion {key}")
        predictions, metrics, alphas = nested_fusion(behavior, msnet)
        feature_name, model_name = key.split("__")
        average = metrics.iloc[-1]
        fusion_predictions[key] = predictions
        fusion_fold_metrics[key] = metrics
        fusion_alphas[key] = alphas
        fusion_metrics_rows.append(
            {
                "key": key,
                "feature_set": feature_name,
                "model": model_name,
                **{name: average[name] for name in ["acc", "sensitivity", "specificity", "auc", "precision", "f1_score"]},
                "threshold_mean": average["threshold"],
                "alpha_mean": alphas["chosen_alpha_msnet"].mean(),
                "behavior_weight_mean": alphas["chosen_behavior_weight"].mean(),
            }
        )
    fusion_metrics = pd.DataFrame(fusion_metrics_rows).sort_values(
        ["auc", "f1_score", "acc"], ascending=False
    ).reset_index(drop=True)
    fusion_metrics.to_csv(fusion_dir / "fusion_metrics_nested_alpha.csv", index=False)
    best_fusion_key = str(fusion_metrics.iloc[0]["key"])
    best_fusion = fusion_predictions[best_fusion_key]
    best_fusion.to_csv(fusion_dir / "best_fusion_prob_val.csv", index=False)
    fusion_fold_metrics[best_fusion_key].to_csv(
        fusion_dir / "best_fusion_foldwise_metrics.csv", index=False
    )
    fusion_alphas[best_fusion_key].to_csv(
        fusion_dir / "best_fusion_alpha_by_fold.csv", index=False
    )

    msnet_average = msnet_metrics.iloc[-1]
    behavior_average = best_behavior_metrics.iloc[-1]
    fusion_average = fusion_fold_metrics[best_fusion_key].iloc[-1]
    comparison = pd.DataFrame(
        [
            {
                "method": "Category MSNet",
                "feature_set": "RINet fixation features",
                "model": "MSNet trained from scratch",
                "alpha_msnet": 1.0,
                "behavior_weight": 0.0,
                **{name: msnet_average[name] for name in ["acc", "sensitivity", "specificity", "auc", "precision", "f1_score"]},
            },
            {
                "method": "Category behavior-only",
                "feature_set": behavior_metrics.iloc[0]["feature_set"],
                "model": behavior_metrics.iloc[0]["model"],
                "alpha_msnet": 0.0,
                "behavior_weight": 1.0,
                **{name: behavior_average[name] for name in ["acc", "sensitivity", "specificity", "auc", "precision", "f1_score"]},
            },
            {
                "method": "Category MSNet + Behavioral Fusion",
                "feature_set": fusion_metrics.iloc[0]["feature_set"],
                "model": fusion_metrics.iloc[0]["model"],
                "alpha_msnet": fusion_metrics.iloc[0]["alpha_mean"],
                "behavior_weight": fusion_metrics.iloc[0]["behavior_weight_mean"],
                **{name: fusion_average[name] for name in ["acc", "sensitivity", "specificity", "auc", "precision", "f1_score"]},
            },
        ]
    )
    comparison.to_csv(output / "final_comparison_metrics.csv", index=False)
    write_json(
        output / "completed.json",
        {
            "category": category,
            "category_name": CATEGORY_DIRS[category],
            "image_count": len(image_sets_from_disk()[category]),
            "best_behavior_key": best_behavior_key,
            "best_fusion_key": best_fusion_key,
        },
    )
    print(f"[{category}] COMPLETE\n{comparison.to_string(index=False)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["all", *CATEGORY_ORDER], required=True)
    args = parser.parse_args()
    categories = CATEGORY_ORDER if args.category == "all" else (args.category,)
    for category in categories:
        run_one(category)


if __name__ == "__main__":
    main()
