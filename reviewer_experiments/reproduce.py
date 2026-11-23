"""Cross-platform entry point for the two 2/1/1 BAMF variants.

Run ``python -m reviewer_experiments.reproduce --help`` from the repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .protocol_211 import FOLDS, validate_subject_map
from .run_211 import METRICS, rank_candidates
from .seed_alpha import sha256
from .train_211 import fixed_threshold_metrics


ROOT = Path(__file__).resolve().parents[1]


def run_module(name: str, *arguments: object) -> None:
    command = [sys.executable, "-m", name, *(str(value) for value in arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def check_inputs(args) -> None:
    dataset = ROOT / "EMS"
    checks = {
        "ems_workbook": dataset / "Train_Valid.xlsx",
        "ems_fixations": dataset / "Train_Valid" / "Fixations",
        "ems_images": dataset / "Images",
        "rinet_features": ROOT / "feature_dict_RINet.npy",
        "ems_submodule": ROOT / "EMS_repo" / "MSNet.py",
        "behavior_features": args.behavior_features,
        "feature_sets": args.feature_sets,
    }
    missing = [name for name, path in checks.items() if not path.exists()]
    if not missing:
        features = pd.read_csv(args.behavior_features)
        validate_subject_map(features)
        groups = json.loads(args.feature_sets.read_text(encoding="utf-8"))
        if len(groups) != 7 or any(not set(cols).issubset(features.columns) for cols in groups.values()):
            raise ValueError("Behavioral features do not contain the seven published feature groups")
    status = {name: {"path": str(path), "exists": path.exists()} for name, path in checks.items()}
    status["missing"] = missing
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))


def prepare_cache(args) -> None:
    if (args.cache_dir / "features_100.npy").exists():
        if not (args.cache_dir / "subjects.csv").exists():
            raise ValueError("Cache features exist without subjects.csv")
        validate_subject_map(pd.read_csv(args.cache_dir / "subjects.csv"))
        manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["feature_dict_sha256"] != sha256(ROOT / "feature_dict_RINet.npy") or
                manifest["workbook_sha256"] != sha256(ROOT / "EMS/Train_Valid.xlsx")):
            raise ValueError("Cached MSNet inputs differ from the current EMS/RINet files")
        print(f"Using existing cache: {args.cache_dir}")
        return
    run_module("reviewer_experiments.prepare_cache", "--project-root", ROOT,
               "--output", args.cache_dir)


def train_msnet(args) -> None:
    for fold in FOLDS:
        run_module("reviewer_experiments.train_211", "--cache-dir", args.cache_dir,
                   "--behavior-features", args.behavior_features,
                   "--feature-sets", args.feature_sets,
                   "--output-root", args.msnet_root,
                   "--eval-fold", fold, "--epochs", args.epochs,
                   "--batch-size", args.batch_size, "--seed", args.seed)


def evaluate(args) -> None:
    run_module("reviewer_experiments.run_211", "--msnet-root", args.msnet_root,
               "--behavior-features", args.behavior_features,
               "--feature-sets", args.feature_sets,
               "--output-root", args.output_root,
               "--variant", args.variant, "--eval-fold", "all", "--seed", args.seed)


def verify(args) -> None:
    features = pd.read_csv(args.behavior_features).set_index("subject_int")
    groups = json.loads(args.feature_sets.read_text(encoding="utf-8"))
    variants = ("fixed", "joint") if args.variant == "both" else (args.variant,)
    for variant in variants:
        all_ids = []
        rows = []
        for fold in FOLDS:
            folder = args.output_root / variant / fold
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            completed = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
            if (manifest["variant"] != variant or manifest["eval_fold"] != fold or
                    manifest["source_sha256"] != sha256(ROOT / "reviewer_experiments/run_211.py") or
                    manifest["behavior_sha256"] != sha256(args.behavior_features) or
                    manifest["feature_sets_sha256"] != sha256(args.feature_sets) or
                    manifest["msnet_checkpoint_sha256"] != sha256(args.msnet_root / fold / "final_model.pt")):
                raise ValueError(f"Manifest differs from current inputs: {variant}/{fold}")
            for key, name in (("manifest_sha256", "manifest.json"),
                              ("behavior_model_sha256", "behavior_model.joblib"),
                              ("selection_candidates_sha256", "selection_candidates.csv")):
                if completed[key] != sha256(folder / name):
                    raise ValueError(f"Saved artifact changed: {folder / name}")
            candidates = pd.read_csv(folder / "selection_candidates.csv")
            if len(candidates) != (21 if variant == "fixed" else 441):
                raise ValueError(f"Wrong candidate count: {variant}/{fold}")
            pd.testing.assert_frame_equal(candidates, rank_candidates(candidates), check_exact=False)
            chosen = candidates.iloc[0]
            selection = pd.read_csv(folder / "selection_predictions.csv")
            evaluation = pd.read_csv(folder / "evaluation_predictions.csv")
            result = pd.read_csv(folder / "evaluation_metrics.csv").iloc[0]
            if (len(selection), len(evaluation)) != (40, 40) or set(selection.set_name) != {manifest["selection_fold"]} or set(evaluation.set_name) != {fold}:
                raise ValueError(f"Wrong subject roles: {variant}/{fold}")
            if set(selection.subject_int) & set(evaluation.subject_int):
                raise ValueError(f"Selection and evaluation subjects overlap: {variant}/{fold}")
            if (str(chosen.feature_set), str(chosen.model), float(chosen.alpha_msnet)) != (
                    result.feature_set, result.model, float(result.alpha_msnet)):
                raise ValueError(f"Reported winner differs from candidate ranking: {variant}/{fold}")
            model = joblib.load(folder / "behavior_model.joblib")
            columns = groups[result.feature_set]
            for part in (selection, evaluation):
                scores = model.predict_proba(features.loc[part.subject_int, columns])[:, 1]
                if not np.allclose(scores, part.behavior_prob, rtol=0, atol=1e-10):
                    raise ValueError(f"Saved model does not reproduce behavioral scores: {variant}/{fold}")
                fused = result.alpha_msnet * part.msnet_prob + (1 - result.alpha_msnet) * scores
                if not np.allclose(fused, part.fused_prob, rtol=0, atol=1e-10):
                    raise ValueError(f"Fusion scores differ: {variant}/{fold}")
            metrics = fixed_threshold_metrics(evaluation.label, evaluation.fused_prob, float(result.threshold))
            if any(not np.isclose(metrics[name], result[name], rtol=0, atol=1e-10) for name in METRICS):
                raise ValueError(f"Evaluation metric differs: {variant}/{fold}")
            if variant == "fixed":
                old = pd.read_csv(args.msnet_root / fold / "evaluation_predictions.csv")
                aligned = evaluation[["subject_int", "fused_prob"]].merge(
                    old[["subject_int", "fused_prob"]], on="subject_int", validate="one_to_one")
                if len(aligned) != 40 or not np.allclose(aligned.fused_prob_x, aligned.fused_prob_y, rtol=0, atol=1e-10):
                    raise ValueError(f"Fixed protocol does not reproduce original run: {fold}")
            all_ids.extend(evaluation.subject_int.tolist())
            rows.append(result)
        if len(all_ids) != 160 or len(set(all_ids)) != 160:
            raise ValueError(f"Expected 160 distinct evaluated people in {variant}")
        summary = pd.read_csv(args.output_root / variant / "summary_metrics.csv", index_col=0)["mean"]
        reported = pd.DataFrame(rows)[list(METRICS)].mean()
        if any(not np.isclose(summary[name], reported[name], rtol=0, atol=1e-10) for name in METRICS):
            raise ValueError(f"Summary differs from fold metrics in {variant}")
        print(f"Verified {variant}: 4 folds, 160 distinct evaluation subjects, "
              f"AUC={summary['auc']:.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = ROOT / "outputs" / "improve_behavioral"
    parser.add_argument("stage", choices=["check", "prepare-cache", "train-msnet", "evaluate", "verify", "all"])
    parser.add_argument("--behavior-features", type=Path,
                        default=baseline / "subject_behavior_features.csv")
    parser.add_argument("--feature-sets", type=Path, default=baseline / "feature_sets_clean.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "work/reviewer_experiments/cache")
    parser.add_argument("--msnet-root", type=Path, default=ROOT / "work/reproduction_211/msnet")
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/reproduction_211/evaluation")
    parser.add_argument("--variant", choices=["fixed", "joint", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.stage in ("check", "all"):
        check_inputs(args)
    if args.stage in ("prepare-cache", "all"):
        prepare_cache(args)
    if args.stage in ("train-msnet", "all"):
        train_msnet(args)
    if args.stage in ("evaluate", "all"):
        evaluate(args)
    if args.stage in ("verify", "all"):
        verify(args)


if __name__ == "__main__":
    main()
