from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CATEGORY_DIRS,
    CATEGORY_ORDER,
    DATASET_DIR,
    EXPERIMENT_ROOT,
    FOLDS,
    FULL_RESULTS_DIR,
    compute_foldwise_metrics,
    write_json,
)
from behavior import (
    META_COLUMNS,
    aggregate_category_features,
    behavior_cv,
    load_msnet_predictions,
)


ALPHA_MSNET = 0.8
BEHAVIOR_WEIGHT = 0.2
FEATURE_SET = "pupil_scanpath"
CLASSIFIER = "hgb"
OUTPUT_ROOT = EXPERIMENT_ROOT.parent / "fixed_original_protocol"
METRIC_COLUMNS = ["acc", "sensitivity", "specificity", "precision", "f1_score", "auc"]


def create_mapping() -> pd.DataFrame:
    rows: list[dict] = []
    for slug, category_name in CATEGORY_DIRS.items():
        folder = DATASET_DIR / "Images" / category_name
        for image in sorted(path for path in folder.iterdir() if path.is_file()):
            rows.append({"image": image.name, "image_stem": image.stem, "category": slug,
                         "category_name": category_name})
    mapping = pd.DataFrame(rows)
    if len(mapping) != 100 or mapping["image_stem"].nunique() != 100:
        raise RuntimeError("Stimulus-category mapping must contain 100 unique images")
    mapping.to_csv(OUTPUT_ROOT / "stimulus_category_mapping.csv", index=False)
    return mapping


def add_fold_decisions(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    thresholds = metrics[metrics["set_name"] != "average"].set_index("set_name")["threshold"]
    result["threshold"] = result["set_name"].map(thresholds).astype(float)
    result["predicted_label"] = (result["p_fusion"] >= result["threshold"]).astype(int)
    return result


def run_category(category: str, image_count: int) -> tuple[dict, pd.DataFrame]:
    category_output = OUTPUT_ROOT / category
    category_output.mkdir(parents=True, exist_ok=True)

    subject, feature_sets = aggregate_category_features(category)
    columns = feature_sets[FEATURE_SET]
    if not columns or any(column.startswith("duration_") for column in columns):
        raise RuntimeError(f"Invalid Pupil+Scanpath feature set for {category}")
    subject.to_csv(category_output / "subject_behavior_features.csv", index=False)
    write_json(category_output / "feature_columns.json", columns)

    # Refit exactly one behavioral model per fold using the frozen paper configuration.
    behavior = behavior_cv(subject, columns)
    msnet = load_msnet_predictions(category)
    behavior_merge = behavior[["subject_int", "set_name", "label", "behavior_prob"]].rename(
        columns={"label": "behavior_label"}
    )
    merged = msnet.merge(
        behavior_merge,
        on=["subject_int", "set_name"], how="inner", validate="one_to_one",
    )
    if len(merged) != 160 or merged["subject_int"].nunique() != 160:
        raise RuntimeError(f"Expected 160 aligned OOF predictions for {category}")
    if not np.array_equal(merged["label"].astype(int), merged.pop("behavior_label").astype(int)):
        raise RuntimeError(f"Label alignment failed for {category}")

    merged["category"] = category
    merged["category_name"] = CATEGORY_DIRS[category]
    merged["p_msnet"] = merged.pop("msnet_prob")
    merged["p_behavior"] = merged.pop("behavior_prob")
    merged["p_fusion"] = ALPHA_MSNET * merged["p_msnet"] + BEHAVIOR_WEIGHT * merged["p_behavior"]

    msnet_for_metrics = merged.rename(columns={"p_msnet": "prob"})
    behavior_for_metrics = merged.rename(columns={"p_behavior": "prob"})
    fusion_for_metrics = merged.rename(columns={"p_fusion": "prob"})
    msnet_metrics = compute_foldwise_metrics(msnet_for_metrics, "prob")
    behavior_metrics = compute_foldwise_metrics(behavior_for_metrics, "prob")
    fusion_metrics = compute_foldwise_metrics(fusion_for_metrics, "prob")
    msnet_metrics.to_csv(category_output / "msnet_foldwise_metrics.csv", index=False)
    behavior_metrics.to_csv(category_output / "behavior_foldwise_metrics.csv", index=False)
    fusion_metrics.to_csv(category_output / "fusion_foldwise_metrics.csv", index=False)

    decisions = add_fold_decisions(merged, fusion_metrics)
    decisions["subject_id"] = decisions["subject_int"].map(lambda value: f"{int(value):03d}")
    decisions = decisions.rename(columns={"label": "true_label", "set_name": "fold"})
    prediction_columns = [
        "subject_id", "subject_int", "true_label", "fold", "category", "category_name",
        "p_msnet", "p_behavior", "p_fusion", "threshold", "predicted_label",
    ]
    decisions[prediction_columns].sort_values(["fold", "subject_int"]).to_csv(
        category_output / "subject_predictions.csv", index=False
    )

    average = fusion_metrics.iloc[-1]
    fold_only = fusion_metrics[fusion_metrics["set_name"] != "average"]
    summary = {
        "stimulus_set": CATEGORY_DIRS[category],
        "category": category,
        "image_count": image_count,
        "feature_set": "Pupil + Scanpath",
        "classifier": "HistGradientBoosting",
        "alpha_msnet": ALPHA_MSNET,
        "behavior_weight": BEHAVIOR_WEIGHT,
        **{name: float(average[name]) for name in METRIC_COLUMNS},
        "auc_std_across_folds": float(fold_only["auc"].std(ddof=1)),
        "auc_range_across_folds": float(fold_only["auc"].max() - fold_only["auc"].min()),
    }
    pd.DataFrame([summary]).to_csv(category_output / "category_summary.csv", index=False)
    return summary, fusion_metrics


def baseline_summary() -> tuple[dict, pd.DataFrame]:
    comparison = pd.read_csv(FULL_RESULTS_DIR / "final_comparison_metrics.csv")
    row = comparison[comparison["method"] == "MSNet + Behavioral Fusion"].iloc[0]
    fold_metrics = pd.read_csv(FULL_RESULTS_DIR / "best_fusion_foldwise_metrics.csv")
    fold_only = fold_metrics[fold_metrics["set_name"] != "average"]
    summary = {
        "stimulus_set": "All",
        "category": "all",
        "image_count": 100,
        "feature_set": "Pupil + Scanpath",
        "classifier": "HistGradientBoosting",
        "alpha_msnet": ALPHA_MSNET,
        "behavior_weight": BEHAVIOR_WEIGHT,
        **{name: float(row[name]) for name in METRIC_COLUMNS},
        "auc_std_across_folds": float(fold_only["auc"].std(ddof=1)),
        "auc_range_across_folds": float(fold_only["auc"].max() - fold_only["auc"].min()),
    }
    return summary, fold_metrics


def write_report(results: pd.DataFrame, winners: pd.DataFrame, path: Path) -> None:
    category_results = results[results["category"] != "all"]
    auc_winner = category_results.loc[category_results["auc"].idxmax()]
    sz_winner = category_results.loc[category_results["sensitivity"].idxmax()]
    hc_winner = category_results.loc[category_results["specificity"].idxmax()]
    stable = category_results.loc[category_results["auc_std_across_folds"].idxmin()]
    table_columns = ["stimulus_set", "image_count", "acc", "sensitivity", "specificity", "f1_score", "auc"]
    table_lines = [
        "| Stimulus set | # Images | Accuracy | Sensitivity | Specificity | F1 | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results[table_columns].iterrows():
        table_lines.append(
            f"| {row['stimulus_set']} | {int(row['image_count'])} | {row['acc']:.4f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} | "
            f"{row['f1_score']:.4f} | {row['auc']:.4f} |"
        )
    lines = [
        "# Stimulus category results",
        "",
        *table_lines,
        "",
        f"- Highest AUC: {auc_winner['stimulus_set']} ({auc_winner['auc']:.4f}).",
        f"- Highest sensitivity: {sz_winner['stimulus_set']} ({sz_winner['sensitivity']:.4f}).",
        f"- Highest specificity: {hc_winner['stimulus_set']} ({hc_winner['specificity']:.4f}).",
        f"- Lowest fold AUC SD: {stable['stimulus_set']} ({stable['auc_std_across_folds']:.4f}).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(results: pd.DataFrame, path: Path) -> None:
    frame = results.set_index("stimulus_set")[["auc", "sensitivity", "specificity"]]
    ax = frame.plot(kind="bar", figsize=(10, 6), ylim=(0.6, 1.0), width=0.75)
    ax.set_title("EMS fixed protocol: MSNet + Pupil/Scanpath + HGB, alpha=0.8")
    ax.set_ylabel("Mean four-fold metric")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["all", *CATEGORY_ORDER], default="all")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mapping = create_mapping()
    counts = mapping.groupby("category").size().to_dict()
    categories = CATEGORY_ORDER if args.category == "all" else (args.category,)

    summaries: list[dict] = []
    fold_tables: dict[str, pd.DataFrame] = {}
    for category in categories:
        print(f"[{category}] fitting fixed Pupil+Scanpath/HGB four-fold OOF")
        summary, folds = run_category(category, int(counts[category]))
        summaries.append(summary)
        fold_tables[CATEGORY_DIRS[category]] = folds
        print(f"[{category}] fusion AUC={summary['auc']:.4f}")

    if args.category != "all":
        return

    baseline, baseline_folds = baseline_summary()
    ordered = pd.DataFrame([baseline, *summaries])
    for metric in METRIC_COLUMNS:
        ordered[f"delta_{metric}_vs_all100"] = ordered[metric] - baseline[metric]
    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(summary_dir / "final_results_with_baseline.csv", index=False)

    fold_auc = pd.DataFrame({"fold": FOLDS})
    baseline_indexed = baseline_folds.set_index("set_name")
    fold_auc["All"] = [baseline_indexed.loc[fold, "auc"] for fold in FOLDS]
    for name, metrics in fold_tables.items():
        indexed = metrics.set_index("set_name")
        fold_auc[name] = [indexed.loc[fold, "auc"] for fold in FOLDS]
    fold_auc.loc[len(fold_auc)] = ["Mean", *[fold_auc[column].mean() for column in fold_auc.columns[1:]]]
    fold_auc.to_csv(summary_dir / "auc_by_fold.csv", index=False)

    category_only = ordered[ordered["category"] != "all"]
    winners = pd.DataFrame([
        {"criterion": "AUC (overall discrimination)", **category_only.loc[category_only["auc"].idxmax()].to_dict()},
        {"criterion": "Sensitivity (SZ)", **category_only.loc[category_only["sensitivity"].idxmax()].to_dict()},
        {"criterion": "Specificity (HC)", **category_only.loc[category_only["specificity"].idxmax()].to_dict()},
        {"criterion": "Lowest fold AUC SD", **category_only.loc[category_only["auc_std_across_folds"].idxmin()].to_dict()},
        {"criterion": "Smallest AUC drop vs All-100", **category_only.loc[category_only["delta_auc_vs_all100"].idxmax()].to_dict()},
    ])
    winners.to_csv(summary_dir / "best_category_by_criterion.csv", index=False)
    write_report(ordered, winners, summary_dir / "RESULTS_FIXED_PROTOCOL.md")
    make_plot(ordered, summary_dir / "fixed_protocol_metrics.png")
    write_json(OUTPUT_ROOT / "protocol.json", {
        "subjects": 160,
        "folds": list(FOLDS),
        "feature_set": "Pupil + Scanpath",
        "classifier": "HistGradientBoosting",
        "hgb_config": {"max_iter": 300, "learning_rate": 0.03, "max_leaf_nodes": 15,
                       "l2_regularization": 0.01, "random_state": 42},
        "alpha_msnet": ALPHA_MSNET,
        "behavior_weight": BEHAVIOR_WEIGHT,
        "thresholding": "OpenCV Otsu independently within each validation fold",
        "metric_aggregation": "unweighted mean across Set_0..Set_3",
        "category_image_counts": counts,
    })
    print("\n" + ordered[["stimulus_set", "image_count", *METRIC_COLUMNS]].to_string(index=False))
    print(f"\nResults written to {summary_dir}")


if __name__ == "__main__":
    main()
