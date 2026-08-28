from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import CATEGORY_DIRS, CATEGORY_ORDER, EXPERIMENT_ROOT, FOLDS, FULL_RESULTS_DIR


METHOD_FILES = {
    "Category MSNet": ("msnet", "msnet_prob"),
    "Category behavior-only": ("behavior", "behavior_prob"),
    "Category MSNet + Behavioral Fusion": ("fusion", "fused_prob"),
}


def prediction_path(category: str, method: str) -> Path:
    kind, _ = METHOD_FILES[method]
    if kind == "msnet":
        return EXPERIMENT_ROOT / category / "all_msnet_prob_val.csv"
    if kind == "behavior":
        return EXPERIMENT_ROOT / category / "behavior" / "best_behavior_prob_val.csv"
    return EXPERIMENT_ROOT / category / "fusion" / "best_fusion_prob_val.csv"


def collect_comparison() -> pd.DataFrame:
    image_counts = {slug: len(list((FULL_RESULTS_DIR.parent / "EMS" / "Images" / dirname).glob("*")))
                    for slug, dirname in CATEGORY_DIRS.items()}
    rows = []
    for category in CATEGORY_ORDER:
        frame = pd.read_csv(EXPERIMENT_ROOT / category / "final_comparison_metrics.csv")
        frame.insert(0, "category", category)
        frame.insert(1, "category_name", CATEGORY_DIRS[category])
        frame.insert(2, "image_count", image_counts[category])
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def paired_bootstrap_fusion(iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    for category in CATEGORY_ORDER:
        path = prediction_path(category, "Category MSNet + Behavioral Fusion")
        frame = pd.read_csv(path)[["subject_int", "set_name", "label", "fused_prob"]]
        predictions[category] = frame.sort_values("subject_int").reset_index(drop=True)

    reference = predictions[CATEGORY_ORDER[0]][["subject_int", "set_name", "label"]]
    for category, frame in predictions.items():
        if not frame[["subject_int", "set_name", "label"]].equals(reference):
            raise RuntimeError(f"OOF subjects/folds do not align for {category}")

    rng = np.random.default_rng(seed)
    auc_samples = {category: np.empty(iterations, dtype=float) for category in CATEGORY_ORDER}
    strata = [
        reference.index[(reference["set_name"] == fold) & (reference["label"] == label)].to_numpy()
        for fold in FOLDS for label in (0, 1)
    ]
    for iteration in range(iterations):
        sampled = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])
        sample_meta = reference.loc[sampled]
        for category, frame in predictions.items():
            fold_aucs = []
            for fold in FOLDS:
                mask = sample_meta["set_name"].to_numpy() == fold
                indices = sampled[mask]
                fold_aucs.append(roc_auc_score(frame.loc[indices, "label"], frame.loc[indices, "fused_prob"]))
            auc_samples[category][iteration] = float(np.mean(fold_aucs))

    ci_rows = []
    for category, values in auc_samples.items():
        ci_rows.append({
            "category": category,
            "bootstrap_mean_auc": values.mean(),
            "ci95_low": np.quantile(values, 0.025),
            "ci95_high": np.quantile(values, 0.975),
        })
    pair_rows = []
    for index, left in enumerate(CATEGORY_ORDER):
        for right in CATEGORY_ORDER[index + 1:]:
            difference = auc_samples[left] - auc_samples[right]
            pair_rows.append({
                "category_a": left,
                "category_b": right,
                "mean_auc_difference_a_minus_b": difference.mean(),
                "ci95_low": np.quantile(difference, 0.025),
                "ci95_high": np.quantile(difference, 0.975),
                "probability_a_gt_b": np.mean(difference > 0),
            })
    return pd.DataFrame(ci_rows), pd.DataFrame(pair_rows)


def make_plot(comparison: pd.DataFrame, path: Path) -> None:
    pivot = comparison.pivot(index="category_name", columns="method", values="auc")
    pivot = pivot.loc[list(CATEGORY_DIRS.values())]
    ax = pivot.plot(kind="bar", figsize=(11, 6), ylim=(0.65, 1.0), width=0.78)
    ax.set_ylabel("Mean 4-fold OOF AUC")
    ax.set_xlabel("")
    ax.set_title("EMS category experiment comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-bootstrap",
        action="store_true",
        help="Reuse existing bootstrap CSVs (only safe when OOF predictions are unchanged).",
    )
    args = parser.parse_args()
    summary = EXPERIMENT_ROOT / "summary"
    summary.mkdir(parents=True, exist_ok=True)

    comparison = collect_comparison()
    comparison.to_csv(summary / "category_comparison_metrics.csv", index=False)
    winners = comparison.loc[comparison.groupby("method")["auc"].idxmax()].copy()
    winners.to_csv(summary / "best_category_by_method.csv", index=False)

    full = pd.read_csv(FULL_RESULTS_DIR / "final_comparison_metrics.csv")
    full.insert(0, "category", "full_ems_reference")
    full.insert(1, "category_name", "Full EMS (Kaggle reference)")
    full.insert(2, "image_count", 100)
    pd.concat([comparison, full], ignore_index=True, sort=False).to_csv(
        summary / "category_plus_full_ems_reference.csv", index=False
    )

    ci_path = summary / "fusion_auc_bootstrap_ci.csv"
    pairs_path = summary / "fusion_auc_pairwise_bootstrap.csv"
    if args.reuse_bootstrap and ci_path.exists() and pairs_path.exists():
        ci = pd.read_csv(ci_path)
        pairs = pd.read_csv(pairs_path)
    else:
        ci, pairs = paired_bootstrap_fusion(args.bootstrap_iterations, args.seed)
        ci.to_csv(ci_path, index=False)
        pairs.to_csv(pairs_path, index=False)
    make_plot(comparison, summary / "category_auc_comparison.png")
    print(winners[["method", "category_name", "image_count", "acc", "auc", "f1_score"]].to_string(index=False))
    print(f"\nSummary written to {summary}")


if __name__ == "__main__":
    main()
