"""Generate the vector figures requested in the supervisor's second review.

The script rebuilds the exact Set_0 pupil--scanpath HGB model used by the
reported four-fold experiment, explains the corrected decision for subject
302 with LIME, and redraws the multi-subject case analysis as decision
margins.  Run with PYTHONPATH pointing to the local analysis dependencies.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import lime.lime_tabular
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "improve_behavioral_final_results"
REPORT = RESULTS / "report_outputs"
REPORT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "msnet": "#2563A6",
    "behavior": "#2E8B57",
    "fusion": "#D97706",
    "hc": "#B42318",
    "sz": "#14804A",
    "grid": "#D9DEE7",
    "ink": "#1F2937",
}


def behavior_model() -> Pipeline:
    """Return the exact behavioral pipeline used for the reported results."""
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


def human_feature_name(feature: str) -> str:
    aggregate_names = {
        "mean": "mean across stimuli",
        "std": "SD across stimuli",
        "median": "median across stimuli",
        "min": "minimum across stimuli",
        "max": "maximum across stimuli",
    }
    aggregate = ""
    base = feature
    for suffix, description in aggregate_names.items():
        token = "_" + suffix
        if feature.endswith(token):
            base = feature[: -len(token)]
            aggregate = description
            break

    base_names = {
        "pupil_delta_step_abs_mean": "mean absolute successive pupil change",
        "pupil_delta_first_abs_mean": "mean absolute pupil change from first fixation",
        "pupil_delta_first_std": "SD of pupil change from first fixation",
        "pupil_delta_first_mean": "mean pupil change from first fixation",
        "pupil_delta_step_std": "SD of successive pupil change",
        "pupil_delta_step_mean": "mean successive pupil change",
        "distance_to_center_mean": "mean distance to screen center",
        "distance_to_center_std": "SD of distance to screen center",
        "distance_to_center_max": "maximum distance to screen center",
        "scanpath_length": "scanpath length",
        "fixation_count": "fixation count",
        "pupil_first_fix": "first-fixation pupil diameter",
        "pupil_first_5_mean": "early pupil mean (first five fixations)",
        "pupil_cv": "pupil coefficient of variation",
        "pupil_iqr": "pupil interquartile range",
        "pupil_mean": "mean pupil diameter",
        "pupil_std": "pupil-diameter SD",
        "pupil_median": "median pupil diameter",
        "pupil_min": "minimum pupil diameter",
        "pupil_max": "maximum pupil diameter",
        "pupil_missing_ratio": "missing-pupil ratio",
        "saccade_amplitude_mean": "mean saccade amplitude",
        "saccade_amplitude_std": "saccade-amplitude SD",
        "saccade_amplitude_median": "median saccade amplitude",
        "saccade_amplitude_max": "maximum saccade amplitude",
        "x_mean": "horizontal gaze mean",
        "x_std": "horizontal gaze SD",
        "y_mean": "vertical gaze mean",
        "y_std": "vertical gaze SD",
        "x_dispersion": "horizontal gaze range",
        "y_dispersion": "vertical gaze range",
        "xy_dispersion_area": "rectangular gaze-dispersion area",
    }
    readable = base_names.get(base, base.replace("_", " "))
    return f"{readable}; {aggregate}" if aggregate else readable


def clean_rule(rule: str, feature_names: list[str]) -> str:
    out = rule
    for feature in sorted(feature_names, key=len, reverse=True):
        if feature in out:
            out = out.replace(feature, human_feature_name(feature))
            break
    return "\n".join(textwrap.wrap(out, width=44))


def create_subject_302_xai() -> None:
    subject = pd.read_csv(RESULTS / "subject_behavior_features.csv")
    feature_sets = json.loads((RESULTS / "feature_sets_clean.json").read_text())
    feature_cols = feature_sets["pupil_scanpath"]

    predictions = pd.read_csv(RESULTS / "best_fusion_prob_val.csv")
    baseline_folds = pd.read_csv(RESULTS / "baseline_msnet_foldwise_metrics.csv")
    fusion_folds = pd.read_csv(RESULTS / "best_fusion_foldwise_metrics.csv")

    case = subject.loc[subject["subject_int"] == 302].iloc[0]
    saved = predictions.loc[predictions["subject_int"] == 302].iloc[0]
    val_set = str(case["set_name"])
    train = subject.loc[subject["set_name"] != val_set].copy()

    model = behavior_model()
    model.fit(train[feature_cols], train["label"].astype(int))
    behavior_prob = float(model.predict_proba(case[feature_cols].to_frame().T)[0, 1])
    if not np.isclose(behavior_prob, float(saved["behavior_prob"]), atol=1e-10):
        raise RuntimeError("Rebuilt behavior probability does not match the saved result")

    alpha = float(saved["alpha_msnet"])
    msnet_prob = float(saved["msnet_prob"])
    fused_prob = alpha * msnet_prob + (1.0 - alpha) * behavior_prob
    if not np.isclose(fused_prob, float(saved["fused_prob"]), atol=1e-10):
        raise RuntimeError("Recomputed fusion probability does not match the saved result")

    msnet_threshold = float(
        baseline_folds.loc[baseline_folds["set_name"] == val_set, "threshold"].iloc[0]
    )
    fusion_threshold = float(
        fusion_folds.loc[fusion_folds["set_name"] == val_set, "threshold"].iloc[0]
    )

    imputer = model.named_steps["imputer"]
    train_behavior = imputer.transform(train[feature_cols])
    case_behavior = imputer.transform(case[feature_cols].to_frame().T)[0]

    msnet = pd.read_csv(RESULTS / "msnet_prob_val.csv")
    msnet_map = msnet.set_index("subject_int")["msnet_prob"]
    train_msnet = msnet_map.loc[train["subject_int"].astype(int)].to_numpy()
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_behavior,
        feature_names=feature_cols,
        class_names=["HC", "SZ"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    def predict_behavior(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        perturbed_behavior = pd.DataFrame(data, columns=feature_cols)
        return model.predict_proba(perturbed_behavior)

    explanation = explainer.explain_instance(
        case_behavior,
        predict_behavior,
        labels=[1],
        num_features=7,
        num_samples=6000,
    )
    local = explanation.as_list(label=1)

    pd.DataFrame(local, columns=["feature_rule", "lime_weight_for_SZ"]).to_csv(
        REPORT / "subject_302_lime_explanation.csv", index=False
    )

    fig = plt.figure(figsize=(10.6, 4.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.86, 1.45])

    ax = fig.add_subplot(grid[0, 0])
    labels = ["MSNet", "Behavior", "Fusion"]
    values = np.array([msnet_prob, behavior_prob, fused_prob]) * 100.0
    bars = ax.barh(
        labels,
        values,
        color=[COLORS["msnet"], COLORS["behavior"], COLORS["fusion"]],
        height=0.56,
    )
    ax.axvline(msnet_threshold * 100.0, color=COLORS["msnet"], ls="--", lw=1.5)
    ax.axvline(fusion_threshold * 100.0, color=COLORS["fusion"], ls=":", lw=1.8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Predicted SZ probability (%)")
    ax.set_title(
        "(a) Decision correction\nSubject 302 | true SZ | MSNet HC → fusion SZ",
        loc="left",
        fontweight="bold",
        fontsize=10,
    )
    ax.grid(axis="x", color=COLORS["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center")
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["msnet"], ls="--", label=f"MSNet threshold {msnet_threshold*100:.2f}%"),
            Line2D([0], [0], color=COLORS["fusion"], ls=":", label=f"Fusion threshold {fusion_threshold*100:.2f}%"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=7.5,
        frameon=False,
    )

    ax = fig.add_subplot(grid[0, 1])
    rules = [clean_rule(rule, feature_cols) for rule, _ in local][::-1]
    weights = np.array([weight for _, weight in local][::-1])
    colors = [COLORS["sz"] if w > 0 else COLORS["hc"] for w in weights]
    ax.barh(np.arange(len(weights)), weights, color=colors, height=0.62)
    ax.axvline(0, color="#5B6573", lw=0.9)
    ax.set_yticks(np.arange(len(weights)), labels=rules)
    ax.set_xlabel("Local contribution to the behavioral SZ score (LIME)")
    ax.set_title("(b) Behavioral evidence around subject 302", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["sz"], lw=7, label="supports SZ"),
            Line2D([0], [0], color=COLORS["hc"], lw=7, label="supports HC"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    for spine in ["top", "right"]:
        fig.axes[0].spines[spine].set_visible(False)
        fig.axes[1].spines[spine].set_visible(False)

    fig.savefig(REPORT / "xai_subject_302.pdf", bbox_inches="tight")
    fig.savefig(REPORT / "xai_subject_302.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def create_case_analysis() -> None:
    corrected = pd.read_csv(REPORT / "examples_corrected_by_fusion.csv", dtype={"subject_id": str})
    remaining = pd.read_csv(REPORT / "examples_remaining_errors.csv", dtype={"subject_id": str})

    def margins(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        msnet = (frame["msnet_prob"] - frame["msnet_threshold"]).to_numpy() * 100.0
        fusion = (frame["fused_prob"] - frame["fusion_threshold"]).to_numpy() * 100.0
        return msnet, fusion

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharex=True, constrained_layout=True)
    panels = [
        (axes[0], corrected, "(a) Errors corrected by fusion", "Corrected"),
        (axes[1], remaining, "(b) Errors remaining after fusion", "Still incorrect"),
    ]

    for ax, frame, title, status in panels:
        msnet_margin, fusion_margin = margins(frame)
        y = np.arange(len(frame))
        ax.axvspan(-60, 0, color="#FBE9E7", alpha=0.55)
        ax.axvspan(0, 60, color="#E7F5EC", alpha=0.55)
        for yi, left, right in zip(y, msnet_margin, fusion_margin):
            ax.plot([left, right], [yi, yi], color="#8A94A3", lw=1.5, zorder=1)
        ax.scatter(msnet_margin, y, s=48, color=COLORS["msnet"], marker="o", label="MSNet", zorder=3)
        ax.scatter(fusion_margin, y, s=55, color=COLORS["fusion"], marker="D", label="Fusion", zorder=3)
        ax.axvline(0, color="#313946", lw=1.1)
        names = [f"{sid.zfill(3)} ({label})" for sid, label in zip(frame["subject_id"], frame["class_name"])]
        ax.set_yticks(y, labels=names)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Decision margin: probability − fold threshold (percentage points)")
        ax.grid(axis="x", color=COLORS["grid"], lw=0.7)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.03, "HC decision", transform=ax.transAxes, color=COLORS["hc"], fontsize=8)
        ax.text(0.98, 0.03, "SZ decision", transform=ax.transAxes, color=COLORS["sz"], fontsize=8, ha="right")
        ax.text(
            0.98,
            0.95,
            status,
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=COLORS["sz"] if status == "Corrected" else COLORS["hc"],
            fontweight="bold",
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    fig.savefig(REPORT / "case_analysis_redesigned.pdf", bbox_inches="tight")
    fig.savefig(REPORT / "case_analysis_redesigned.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def create_alpha_curve() -> None:
    raw = pd.read_csv(REPORT / "alpha_grid_auc_curve.csv")
    data = raw.groupby("alpha_msnet", as_index=False)["auc"].mean()
    fig, ax = plt.subplots(figsize=(5.4, 3.25), constrained_layout=True)
    x = data["alpha_msnet"].to_numpy()
    y = data["auc"].to_numpy() * 100.0
    ax.plot(x, y, color=COLORS["msnet"], lw=2.0, marker="o", ms=4.0)
    best = int(np.argmax(y))
    ax.scatter([x[best]], [y[best]], s=85, color=COLORS["fusion"], zorder=4)
    ax.axvline(x[best], color=COLORS["fusion"], ls="--", lw=1.2)
    ax.annotate(
        f"best: α={x[best]:.1f}, AUC={y[best]:.2f}%",
        xy=(x[best], y[best]),
        xytext=(-88, -28),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=COLORS["ink"]),
        fontsize=8.5,
    )
    ax.set_xlabel("MSNet weight α")
    ax.set_ylabel("Mean four-fold AUC (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(color=COLORS["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(REPORT / "alpha_grid_auc_curve_revised.pdf", bbox_inches="tight")
    fig.savefig(REPORT / "alpha_grid_auc_curve_revised.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    create_subject_302_xai()
    create_case_analysis()
    create_alpha_curve()
    print("Generated:")
    for name in [
        "xai_subject_302.pdf",
        "xai_subject_302.png",
        "subject_302_lime_explanation.csv",
        "case_analysis_redesigned.pdf",
        "case_analysis_redesigned.png",
        "alpha_grid_auc_curve_revised.pdf",
        "alpha_grid_auc_curve_revised.png",
    ]:
        print(REPORT / name)
