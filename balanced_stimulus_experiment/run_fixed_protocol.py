from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from common import (
    ALPHA_MSNET,
    BEHAVIOR_WEIGHT,
    FOLDS,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    STIMULUS_COUNTS,
    load_manifest,
    result_dir,
    write_json,
)


sys.path.insert(0, str(PROJECT_ROOT))
from category_experiments.common import compute_foldwise_metrics


META_COLUMNS = ["subject_int", "subject_id", "set_name", "label"]


def behavior_cv(subject: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    predictions = []
    for fold in FOLDS:
        train = subject[subject["set_name"] != fold]
        validation = subject[subject["set_name"] == fold].copy()
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                l2_regularization=0.01, random_state=42,
            )),
        ])
        model.fit(train[columns], train["label"].astype(int))
        validation["behavior_prob"] = model.predict_proba(validation[columns])[:, 1]
        predictions.append(validation[META_COLUMNS + ["behavior_prob"]])
    return pd.concat(predictions, ignore_index=True).sort_values(["set_name", "subject_int"])


BASELINE_METRICS = {
    "stimulus_count": 100,
    "images_per_category": "unbalanced",
    "acc": 0.84375,
    "sensitivity": 0.8463068181818183,
    "specificity": 0.8361111111111111,
    "precision": 0.8403217697335345,
    "f1_score": 0.840531727044885,
    "auc": 0.9502533143939393,
}
METRICS = ("acc", "sensitivity", "specificity", "precision", "f1_score", "auc")


def aggregate_selected_features(stimulus_count: int) -> tuple[pd.DataFrame, list[str]]:
    source = PROJECT_ROOT / "improve_behavioral_final_results" / "stim_behavior_features.csv"
    stimulus = pd.read_csv(source)
    selected = set(load_manifest(stimulus_count)["image_stem"])
    image_stems = stimulus["image"].astype(str).map(lambda value: Path(value).stem)
    stimulus = stimulus[image_stems.isin(selected)].copy()
    if set(stimulus["image"].astype(str).map(lambda value: Path(value).stem)) != selected:
        raise RuntimeError(f"Behavior table does not cover manifest {stimulus_count}")

    stimulus_meta = ["subject_int", "subject_id", "set_name", "label", "image"]
    numeric = [column for column in stimulus.columns if column not in stimulus_meta]
    aggregated = stimulus.groupby("subject_int")[numeric].agg(["mean", "std", "median", "min", "max"])
    aggregated.columns = [f"{column}_{function}" for column, function in aggregated.columns]
    aggregated = aggregated.reset_index()
    metadata = stimulus[META_COLUMNS].drop_duplicates("subject_int")
    valid_images = stimulus.groupby("subject_int")["image"].nunique().reset_index(name="num_valid_images")
    total_fixations = stimulus.groupby("subject_int")["fixation_count"].sum().reset_index(name="num_total_fixations")
    mean_fixations = stimulus.groupby("subject_int")["fixation_count"].mean().reset_index(name="mean_fixations_per_image")
    subject = (
        metadata.merge(aggregated, on="subject_int", how="left")
        .merge(valid_images, on="subject_int", how="left")
        .merge(total_fixations, on="subject_int", how="left")
        .merge(mean_fixations, on="subject_int", how="left")
        .sort_values("subject_int").reset_index(drop=True)
    )
    if len(subject) != 160:
        raise RuntimeError(f"Expected 160 subjects, found {len(subject)}")
    all_features = [column for column in subject if column not in META_COLUMNS]
    pupil = sorted(column for column in all_features if column.startswith("pupil_"))
    scanpath_bases = [
        "fixation_count", "x_mean", "y_mean", "x_std", "y_std", "x_dispersion",
        "y_dispersion", "xy_dispersion_area", "distance_to_center_mean",
        "distance_to_center_std", "distance_to_center_max", "scanpath_length",
        "saccade_amplitude_mean", "saccade_amplitude_std", "saccade_amplitude_median",
        "saccade_amplitude_max", "num_valid_images", "num_total_fixations",
        "mean_fixations_per_image",
    ]
    scanpath = sorted(
        column for column in all_features
        if any(column == base or column.startswith(base + "_") for base in scanpath_bases)
    )
    features = sorted(set(pupil + scanpath))
    if len(features) != 173 or any(column.startswith("duration_") for column in features):
        raise RuntimeError(f"Expected 173 Pupil+Scanpath features, found {len(features)}")
    subject["num_manifest_images"] = stimulus_count
    return subject, features


def load_msnet_predictions(stimulus_count: int) -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        path = result_dir(stimulus_count) / "msnet" / fold / "prob_val.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 160 or result["subject_int"].nunique() != 160:
        raise RuntimeError(f"Invalid MSNet OOF predictions for {stimulus_count}")
    return result


def add_decisions(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    thresholds = metrics[metrics["set_name"] != "average"].set_index("set_name")["threshold"]
    predictions["threshold"] = predictions["set_name"].map(thresholds).astype(float)
    predictions["predicted_label"] = (predictions["p_fusion"] >= predictions["threshold"]).astype(int)
    return predictions


def run_condition(stimulus_count: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    output = result_dir(stimulus_count)
    subject, features = aggregate_selected_features(stimulus_count)
    behavior = behavior_cv(subject, features)
    msnet = load_msnet_predictions(stimulus_count)
    behavior_merge = behavior[["subject_int", "set_name", "label", "behavior_prob"]].rename(
        columns={"label": "behavior_label"}
    )
    merged = msnet.merge(behavior_merge, on=["subject_int", "set_name"], validate="one_to_one")
    if not np.array_equal(merged["label"].astype(int), merged.pop("behavior_label").astype(int)):
        raise RuntimeError("Behavior/MSNet label mismatch")
    merged["p_msnet"] = merged.pop("msnet_prob")
    merged["p_behavior"] = merged.pop("behavior_prob")
    merged["p_fusion"] = ALPHA_MSNET * merged["p_msnet"] + BEHAVIOR_WEIGHT * merged["p_behavior"]

    component_frames = []
    component_metrics = {}
    for component, probability in (("msnet", "p_msnet"), ("behavior", "p_behavior"), ("fusion", "p_fusion")):
        metrics = compute_foldwise_metrics(merged, probability)
        tagged = metrics.copy()
        tagged.insert(0, "component", component)
        tagged.insert(0, "stimulus_count", stimulus_count)
        component_frames.append(tagged)
        component_metrics[component] = metrics
    all_metrics = pd.concat(component_frames, ignore_index=True)
    all_metrics.to_csv(output / "fold_metrics.csv", index=False)
    write_json(output / "feature_columns.json", features)

    decisions = add_decisions(merged.copy(), component_metrics["fusion"])
    decisions["subject_id"] = decisions["subject_int"].map(lambda value: f"{int(value):03d}")
    decisions = decisions.rename(columns={"set_name": "fold", "label": "true_label"})
    decisions.insert(3, "stimulus_count", stimulus_count)
    columns = ["subject_id", "subject_int", "fold", "stimulus_count", "true_label",
               "p_msnet", "p_behavior", "p_fusion", "threshold", "predicted_label"]
    decisions[columns].sort_values(["fold", "subject_int"]).to_csv(
        output / "subject_predictions.csv", index=False
    )

    average = component_metrics["fusion"].iloc[-1]
    fold_only = component_metrics["fusion"].iloc[:-1]
    summary = {
        "stimulus_count": stimulus_count,
        "images_per_category": stimulus_count // 4,
        **{metric: float(average[metric]) for metric in METRICS},
        "auc_std": float(fold_only["auc"].std(ddof=1)),
        "auc_range": float(fold_only["auc"].max() - fold_only["auc"].min()),
        "delta_auc_vs_baseline": float(average["auc"] - BASELINE_METRICS["auc"]),
    }
    return summary, all_metrics, decisions[columns]


def write_summary(results: pd.DataFrame, fold_metrics: pd.DataFrame) -> None:
    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(summary_dir / "aggregate_metrics.csv", index=False)
    fold_metrics.to_csv(summary_dir / "fold_metrics.csv", index=False)
    fusion = fold_metrics[(fold_metrics["component"] == "fusion") & (fold_metrics["set_name"] != "average")]
    pivot = fusion.pivot(index="set_name", columns="stimulus_count", values="auc")
    pivot.to_csv(summary_dir / "fusion_auc_by_fold.csv")

    plot = results.sort_values("stimulus_count")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metric in ("auc", "sensitivity", "specificity"):
        ax.plot(plot["stimulus_count"], plot[metric], marker="o", linewidth=2, label=metric)
    ax.set_xticks([20, 40, 60, 100])
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("Number of stimuli")
    ax.set_ylabel("Mean four-fold metric")
    ax.set_title("Balanced EMS stimulus-count experiment")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(summary_dir / "metric_vs_stimulus_count.png", dpi=180)
    plt.close(fig)

    subset = results[results["stimulus_count"] < 100].sort_values("stimulus_count")
    closest = subset.loc[subset["delta_auc_vs_baseline"].abs().idxmin()]
    by_count = results.set_index("stimulus_count")
    gain_20_40 = by_count.loc[40, "auc"] - by_count.loc[20, "auc"]
    gain_40_60 = by_count.loc[60, "auc"] - by_count.loc[40, "auc"]
    lines = [
        "# Balanced stimulus-count experiment",
        "",
        "Fixed configuration: MSNet + Pupil/Scanpath + HGB, `alpha_msnet=0.8`.",
        "All-100 is the unchanged Kaggle pretrained four-fold reference.",
        "",
        "| Stimuli | Images/category | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC | Delta AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results.sort_values("stimulus_count").iterrows():
        images = row["images_per_category"]
        lines.append(
            f"| {int(row['stimulus_count'])} | {images} | {row['acc']:.4f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} | {row['precision']:.4f} | "
            f"{row['f1_score']:.4f} | {row['auc']:.4f} | {row['delta_auc_vs_baseline']:.4f} |"
        )
    lines.extend([
        "",
        f"Closest subset to baseline by absolute AUC delta: {int(closest['stimulus_count'])} stimuli "
        f"(AUC={closest['auc']:.4f}, delta={closest['delta_auc_vs_baseline']:.4f}).",
        "",
        f"- AUC gain from 20 to 40 stimuli: {gain_20_40:+.4f}.",
        f"- AUC gain from 40 to 60 stimuli: {gain_40_60:+.4f}.",
        f"- At 60 stimuli, accuracy differs from baseline by "
        f"{by_count.loc[60, 'acc'] - by_count.loc[100, 'acc']:+.4f} and F1 by "
        f"{by_count.loc[60, 'f1_score'] - by_count.loc[100, 'f1_score']:+.4f}.",
        f"- At 60 stimuli, sensitivity differs by "
        f"{by_count.loc[60, 'sensitivity'] - by_count.loc[100, 'sensitivity']:+.4f}, while "
        f"specificity differs by {by_count.loc[60, 'specificity'] - by_count.loc[100, 'specificity']:+.4f}.",
        "",
        "This single seed does not quantify uncertainty from stimulus sampling.",
    ])
    (summary_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    summaries, metric_frames, prediction_frames = [], [], []
    for count in STIMULUS_COUNTS:
        print(f"[{count}] fitting fixed Pupil+Scanpath/HGB and alpha=0.8")
        summary, metrics, predictions = run_condition(count)
        summaries.append(summary)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
    baseline = {
        **BASELINE_METRICS,
        "auc_std": float("nan"), "auc_range": float("nan"), "delta_auc_vs_baseline": 0.0,
    }
    results = pd.DataFrame([*summaries, baseline])
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(summary_dir / "subject_predictions.csv", index=False)
    write_summary(results, all_metrics)
    write_json(OUTPUT_ROOT / "protocol.json", {
        "source": "message2.txt", "seed": 42, "stimulus_counts": list(STIMULUS_COUNTS),
        "nested": True, "subjects": 160, "folds": list(FOLDS),
        "feature_set": "Pupil + Scanpath", "classifier": "HistGradientBoosting",
        "hgb_config": {"max_iter": 300, "learning_rate": 0.03, "max_leaf_nodes": 15,
                       "l2_regularization": 0.01, "random_state": 42},
        "alpha_msnet": ALPHA_MSNET, "behavior_weight": BEHAVIOR_WEIGHT,
        "thresholding": "OpenCV Otsu independently within each validation fold",
        "baseline_auc": BASELINE_METRICS["auc"],
    })
    print(results.sort_values("stimulus_count").to_string(index=False))


if __name__ == "__main__":
    main()
