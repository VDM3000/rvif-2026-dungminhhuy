"""Run the seed and fusion experiments."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "outputs/improve_behavioral"
FOLDS = tuple(f"Set_{i}" for i in range(4))


def run(*args):
    command = [sys.executable, "-u", *map(str, args)]
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "seeds", "early-31", "msnet-211",
                                         "late-211", "early-211", "all"))
    parser.add_argument("--output", type=Path, default=ROOT / "work/paper_reproduction")
    parser.add_argument("--cache", type=Path, default=ROOT / "work/ems_211_cache")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--seed-snapshot", type=Path, help="Optional alternate input for the seed runs.")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "model_parameters")
    parser.add_argument("--msnet-211", type=Path, help="Reuse four saved batch-8 runs for the fusion stages.")
    args = parser.parse_args()
    output = args.output.resolve()
    cache = args.cache.resolve()
    baseline = args.baseline.resolve()
    msnet = args.msnet_211.resolve() if args.msnet_211 else output / "msnet_211"
    behavior = ("--behavior-features", baseline / "subject_behavior_features.csv",
                "--feature-sets", baseline / "feature_sets_clean.json")

    if args.stage in ("prepare", "all") and not (cache / "manifest.json").is_file():
        run("EMS_repo/data_process/split_fix.py", "--dataset_dir", "EMS")
        run("-m", "reviewer_experiments.prepare_cache", "--output", cache)
    if args.stage in ("seeds", "all"):
        seeds_output = output / "seeds"
        if seeds_output.exists():
            parser.error(f"Seed output already exists: {seeds_output}. Choose a fresh --output.")
        run("-m", "reviewer_experiments.seed_alpha", "--baseline", args.seed_snapshot.resolve() if args.seed_snapshot else baseline,
            "--output", seeds_output)
    if args.stage in ("early-31", "all"):
        run("-m", "reviewer_experiments.early_31", "--head-type", "linear", "--cache-dir", cache,
            "--checkpoint-root", args.pretrained.resolve(), *behavior,
            "--historical-msnet", baseline / "msnet_prob_val.csv",
            "--historical-late", baseline / "best_fusion_prob_val.csv",
            "--output-root", output / "early_31")
    if args.stage == "msnet-211" or (args.stage == "all" and args.msnet_211 is None):
        for fold in FOLDS:
            run("-m", "reviewer_experiments.train_211", "--cache-dir", cache, *behavior,
                "--output-root", msnet, "--eval-fold", fold, "--ems-checkpoint-selection", "--require-cuda")
    if args.stage in ("late-211", "early-211", "all"):
        cache_sha = None
        if args.stage in ("early-211", "all"):
            with (cache / "features_100.npy").open("rb") as cache_file:
                cache_sha = hashlib.file_digest(cache_file, "sha256").hexdigest()
        for fold in FOLDS:
            completion = msnet / fold / "completed.json"
            if not completion.is_file():
                parser.error(f"Missing completed MSNet fold: {msnet / fold}. Run msnet-211 first.")
            config = json.loads(completion.read_text(encoding="utf-8"))["config"]
            if (config["batch_size"], config["epochs"], config["seed"], config.get("checkpoint_rule")) != (
                8, 50, 42, "middle-fold Otsu accuracy + AUC; reload best before each next epoch"
            ):
                parser.error(f"Expected batch 8, 50 epochs, seed 42 and tuning-fold checkpoint selection: {completion}.")
            if cache_sha is not None and config["cache_sha256"] != cache_sha:
                parser.error(f"Fixation cache differs from the MSNet training cache in {completion}.")
    if args.stage in ("late-211", "all"):
        for fold in FOLDS:
            if not (output / "fixed" / fold / "completed.json").is_file():
                run("-m", "reviewer_experiments.run_211", "--msnet-root", msnet, *behavior,
                    "--output-root", output, "--variant", "fixed", "--eval-fold", fold)
    if args.stage in ("early-211", "all"):
        early = output / "early_211"
        for fold in FOLDS:
            run("-m", "reviewer_experiments.early_211", "--head-type", "linear", "--cache-dir", cache,
                *behavior, "--late-root", msnet, "--output-root", early, "--eval-fold", fold)
        summary = output / "summary_211"
        if not summary.exists():
            run("-m", "reviewer_experiments.summarize_211", "--workbook", "EMS/Train_Valid.xlsx",
                "--late-root", msnet, "--early-root", early, "--output", summary)


if __name__ == "__main__":
    main()
