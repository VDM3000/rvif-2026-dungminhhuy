"""Recheck saved reviewer results without retraining or overwriting any run.

Run from the repository root with ``python -m tools.audit_reviewer_results
--output work/reviewer_audit``. The report is a retrospective audit, not an
original pre-run manifest. MSNet embeddings are checked as saved artifacts;
this command does not regenerate them from the 11 GiB RINet dictionary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reviewer_experiments.early_211 import head_scores, make_head
from reviewer_experiments.protocol_211 import FOLDS, alpha_candidates, fold_roles, load_subject_map, otsu_threshold, select_alpha
from reviewer_experiments.seed_alpha import alpha_curve, check_subjects, fold_metrics, sha256, summarize_seed_metrics
from reviewer_experiments.summarize_211 import METRICS, audit_evaluations, collect, msnet_from_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    runs = root / "outputs/reviewer_experiments"
    baseline = runs / "baseline_2026-11-22_seed42"
    seed_root = runs / "seed_alpha_final_2026-11-22"
    late_root = runs / "late_211_2026-11-22"
    early_root = runs / "early_211_2026-11-22"
    expected = load_subject_map(root / "EMS/Train_Valid.xlsx")
    behavior_path = baseline / "improve_behavioral/subject_behavior_features.csv"
    sets_path = baseline / "improve_behavioral/feature_sets_clean.json"
    features = pd.read_csv(behavior_path)
    msnet = pd.read_csv(baseline / "improve_behavioral/msnet_prob_val.csv")
    columns = json.loads(sets_path.read_text())["pupil_scanpath"]
    check_subjects(features, msnet, columns)
    audit_evaluations(features, expected)
    report = {"audited_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
              "kind": "retrospective saved-artifact audit; no retraining", "folds": {},
              "text_hash_checks": {},
              "warnings": ["Original 2/1/1 pre-run manifests are incomplete; this audit cannot backdate provenance.",
                           "MSNet embeddings were not regenerated from raw RINet features."]}
    methods = []
    for name, folder in [("Late Fusion 2/1/1", late_root), ("Early Fusion 2/1/1", early_root)]:
        predictions, metrics = collect(folder, expected, name)
        methods.append(metrics)
        report[name] = {"subjects": len(predictions), "unique_subjects": int(predictions.subject_int.nunique())}
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    baseline_metrics = []
    curves = []
    for fold, ax in zip(FOLDS, axes.flat):
        train_folds, calibration_fold, _ = fold_roles(fold)
        late = late_root / fold
        early = early_root / fold
        complete = json.loads((late / "completed.json").read_text())
        early_complete = json.loads((early / "completed.json").read_text())
        checkpoint = late / "final_model.pt"
        assert sha256(checkpoint) == complete["checkpoint_sha256"] == early_complete["msnet_checkpoint_sha256"], fold
        assert sha256(early / "head.pt") == early_complete["head_sha256"], fold
        for saved in (complete["config"], early_complete):
            for path, key in ((behavior_path, "behavior_sha256"), (sets_path, "feature_sets_sha256")):
                # Git forces LF; the original Windows run hashed CRLF text.
                raw = path.read_bytes()
                original_windows_bytes = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                exact = sha256(path) == saved[key]
                assert exact or hashlib.sha256(original_windows_bytes).hexdigest() == saved[key], (fold, key)
                report["text_hash_checks"][path.name] = "exact bytes" if exact else "matches original after LF-to-CRLF conversion"
        history = pd.read_csv(late / "training_history.csv")
        assert history.epoch.tolist() == list(range(1, 51)), fold
        calibration = pd.read_csv(late / "calibration_predictions.csv")
        evaluated = pd.read_csv(late / "evaluation_predictions.csv")
        audit_evaluations(calibration, expected.loc[expected.set_name == calibration_fold])
        choice = select_alpha(calibration.label, calibration.msnet_prob, calibration.behavior_prob)
        for key in ("alpha_msnet", "threshold"):
            assert np.isclose(choice[key], complete["result"][key], rtol=0, atol=1e-12), (fold, key)
        for part in (calibration, evaluated):
            np.testing.assert_allclose(part.fused_prob, choice["alpha_msnet"] * part.msnet_prob +
                                       (1 - choice["alpha_msnet"]) * part.behavior_prob, rtol=0, atol=1e-12)
        curve = alpha_candidates(calibration.label, calibration.msnet_prob, calibration.behavior_prob)
        curve.insert(0, "evaluation_fold", fold)
        curve.insert(1, "calibration_fold", calibration_fold)
        curves.append(curve)
        ax.plot(curve.alpha_msnet, curve.auc, marker=".")
        ax.scatter([choice["alpha_msnet"]], [choice["auc"]], color="red", zorder=3)
        ax.set_title(f"Evaluate {fold}; tune {calibration_fold}; selected alpha={choice['alpha_msnet']:.2f}", fontsize=9)
        ax.set(xlabel="MSNet weight alpha", ylabel="Calibration AUC")
        _, metric = msnet_from_calibration(calibration, evaluated)
        baseline_metrics.append({"method": "MSNet 2/1/1", "fold": fold, **{k: metric[k] for k in METRICS}})

        embedding_frame = pd.read_csv(early / "msnet_embeddings.csv")
        assert embedding_frame.shape == (160, 57) and embedding_frame.subject_int.nunique() == 160, fold
        aligned = expected[["subject_int"]].merge(embedding_frame, validate="one_to_one")
        embeddings = aligned.drop(columns="subject_int").to_numpy(dtype=np.float32)
        behavior = expected[["subject_int"]].merge(features[["subject_int", *columns]], validate="one_to_one")
        train_idx = np.flatnonzero(expected.set_name.isin(train_folds))
        hgb = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03,
                         max_leaf_nodes=15, l2_regularization=0.01, random_state=complete["config"]["seed"]))])
        hgb.fit(behavior.iloc[train_idx][columns], expected.iloc[train_idx].label)
        behavior_prob = pd.Series(hgb.predict_proba(behavior[columns])[:, 1], index=expected.subject_int)
        for part in (calibration, evaluated):
            np.testing.assert_allclose(part.subject_int.map(behavior_prob), part.behavior_prob, rtol=0, atol=1e-12)
        msnet_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        msnet_head = torch.nn.Sequential(torch.nn.Linear(56, 1), torch.nn.Sigmoid())
        msnet_head[0].load_state_dict({"weight": msnet_state["head.1.weight"], "bias": msnet_state["head.1.bias"]})
        with torch.inference_mode():
            embedding_msnet_scores = msnet_head(torch.from_numpy(embeddings)).flatten().numpy()
        embedding_msnet_map = dict(zip(expected.subject_int, embedding_msnet_scores))
        for part in (calibration, evaluated):
            np.testing.assert_allclose(part.subject_int.map(embedding_msnet_map), part.msnet_prob, rtol=0, atol=2e-6)
        prep = joblib.load(early / "preprocessing.joblib")
        assert prep["behavior_columns"] == columns, fold
        imputer = SimpleImputer(strategy="median").fit(behavior.iloc[train_idx][columns])
        np.testing.assert_allclose(imputer.statistics_, prep["imputer"].statistics_, equal_nan=True)
        imputed = imputer.transform(behavior[columns])
        scaler = StandardScaler().fit(imputed[train_idx])
        np.testing.assert_allclose(scaler.mean_, prep["scaler"].mean_)
        np.testing.assert_allclose(scaler.scale_, prep["scaler"].scale_)
        combined = np.concatenate((embeddings, scaler.transform(imputed).astype(np.float32)), axis=1)
        assert combined.shape == (160, 229) and np.isfinite(combined).all(), fold
        head = make_head()
        head.load_state_dict(torch.load(early / "head.pt", map_location="cpu", weights_only=True))
        head.eval()
        scores = head_scores(head, combined, np.arange(160), torch.device("cpu"))
        score_map = dict(zip(expected.subject_int, scores))
        max_error = 0.0
        for filename in ("calibration_predictions.csv", "evaluation_predictions.csv"):
            saved = pd.read_csv(early / filename)
            reproduced = saved.subject_int.map(score_map).to_numpy()
            max_error = max(max_error, float(np.max(np.abs(reproduced - saved.early_prob))))
            np.testing.assert_allclose(reproduced, saved.early_prob, rtol=0, atol=2e-6)
        early_calibration = pd.read_csv(early / "calibration_predictions.csv")
        audit_evaluations(early_calibration, expected.loc[expected.set_name == calibration_fold])
        assert np.isclose(otsu_threshold(early_calibration.early_prob), early_complete["result"]["threshold"]), fold
        early_history = pd.read_csv(early / "training_history.csv")
        best = early_history.iloc[int(np.argmax(early_history.calibration_auc))]
        assert int(best.epoch) == early_complete["result"]["best_epoch"], fold
        report["folds"][fold] = {"train_folds": list(train_folds), "calibration_fold": calibration_fold,
                                "alpha": choice["alpha_msnet"], "msnet_epochs": 50,
                                "early_shape": list(combined.shape), "early_head_max_probability_error": max_error,
                                "checkpoint_hashes_match": True, "preprocessing_fitted_on_train_only": True}
        report["folds"][fold]["hgb_refit_on_80_train_matches_predictions"] = True
        report["folds"][fold]["saved_embeddings_match_msnet_head_scores"] = True
    figure.tight_layout()
    figure.savefig(args.output / "alpha_calibration_211.png", dpi=180)
    plt.close(figure)
    pd.concat(curves).to_csv(args.output / "alpha_calibration_211.csv", index=False)
    recomputed = pd.concat([pd.DataFrame(baseline_metrics), *methods], ignore_index=True)
    summary = recomputed.groupby("method", sort=False)[list(METRICS)].mean()
    saved_summary = pd.read_csv(runs / "summary_211_2026-11-22/summary_metrics.csv").set_index("method")
    np.testing.assert_allclose(summary, saved_summary.loc[summary.index, list(METRICS)], rtol=0, atol=1e-12)
    recomputed.to_csv(args.output / "recomputed_fold_metrics.csv", index=False)
    summary.to_csv(args.output / "recomputed_summary.csv")
    report["mean_auc"] = summary.auc.to_dict()
    seed_rows = []
    reference = None
    for seed in (42, 17, 2026):
        predictions = pd.read_csv(seed_root / f"seed_{seed}/predictions.csv")
        audit_evaluations(predictions, expected)
        scores = predictions[["subject_int", "msnet_prob", "behavior_prob", "fused_prob"]].sort_values("subject_int")
        if reference is not None:
            np.testing.assert_array_equal(scores.to_numpy(), reference.to_numpy())
        reference = scores
        metrics = fold_metrics(predictions, "fused_prob")
        seed_rows.append({"seed": seed, **metrics.iloc[-1][list(METRICS)].to_dict()})
    seed_summary = summarize_seed_metrics(pd.DataFrame(seed_rows))
    saved_seed_summary = pd.read_csv(seed_root / "seed_summary.csv", index_col=0)
    np.testing.assert_allclose(seed_summary, saved_seed_summary.loc[seed_summary.index], rtol=0, atol=1e-12)
    curve = alpha_curve(pd.read_csv(seed_root / "seed_42/predictions.csv"))
    stored_curve = pd.read_csv(seed_root / "alpha_curve_by_fold.csv")
    np.testing.assert_allclose(curve.auc, stored_curve.auc, rtol=0, atol=1e-12)
    report["seeds"] = {"identical_predictions": True, "sample_variance": seed_summary.sample_variance.to_dict(),
                       "negative_fold_alpha_deltas": int((curve.delta_auc_vs_msnet < -1e-12).sum()),
                       "negative_mean_alpha_deltas": int((curve.groupby("alpha_msnet").delta_auc_vs_msnet.mean() < -1e-12).sum())}
    report["status"] = "PASS"
    (args.output / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
