"""Train and evaluate one leakage-separated 2/1/1 MSNet late-fusion fold."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .protocol_211 import alpha_candidates, fold_roles, otsu_threshold, select_alpha, validate_subject_map
from .seed_alpha import sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "EMS_repo"))
from MSNet import Model


class CachedSubjects(Dataset):
    def __init__(self, cache: np.ndarray, metadata: pd.DataFrame, indices: np.ndarray):
        self.cache = cache
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source = int(self.indices[index])
        return (torch.from_numpy(np.array(self.cache[source], copy=True)),
                torch.tensor(float(self.metadata.iloc[source].label), dtype=torch.float32),
                int(self.metadata.iloc[source].subject_int))


def fixed_threshold_metrics(labels, scores, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "acc": float(accuracy_score(labels, predicted)),
            "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "auc": float(roc_auc_score(labels, scores)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "f1_score": float(f1_score(labels, predicted, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_state(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temp)
    temp.replace(path)


def train_msnet(cache, metadata, train_indices, folder: Path, epochs: int, batch_size: int, seed: int, device,
                selection_indices=None):
    seed_everything(seed)
    # The upstream EMS Model defaults to 100 stimuli. This standard 2/1/1 run
    # therefore works with the pinned submodule without the variable-count override.
    model = Model(in_c=1056).to(device)
    optimizer = create_optimizer_v2(model, opt="adamw", lr=5e-4, weight_decay=0.05, eps=1e-8)
    scheduler, _ = create_scheduler_v2(optimizer, sched="cosine", num_epochs=epochs, min_lr=1e-5,
                                       warmup_lr=1e-6, warmup_epochs=5, cooldown_epochs=10)
    criterion = nn.BCELoss()
    final_path = folder / "final_model.pt"
    state_path = folder / "training_state.pt"
    if final_path.exists():
        model.load_state_dict(torch.load(final_path, map_location=device, weights_only=True))
        return model, final_path
    start_epoch = 0
    global_steps = -1
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["next_epoch"])
        global_steps = int(state["global_steps"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"resuming epoch {start_epoch + 1}/{epochs}", flush=True)
    history_path = folder / "training_history.csv"
    history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    history = history[:start_epoch]
    best_path = folder / "best_model.pt"
    best_score = max((row["selection_score"] for row in history), default=0.0) if selection_indices is not None else 0.0
    for epoch in range(start_epoch, epochs):
        if selection_indices is not None and epoch:
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(CachedSubjects(cache, metadata, train_indices), batch_size=batch_size,
                            shuffle=True, num_workers=0, generator=generator)
        model.train()
        losses = []
        for features, labels, _ in loader:
            global_steps += 1
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            probability = model(features, alpha=min(1.0, global_steps / 50.0))
            loss = criterion(probability, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch + 1, "train_bce": float(np.mean(losses)),
                        "lr": float(optimizer.param_groups[0]["lr"])})
        if selection_indices is not None:
            selection = msnet_scores(model, cache, metadata, selection_indices, device,
                                     alpha=min(1.0, global_steps / 50.0))
            metrics = fixed_threshold_metrics(selection.label, selection.msnet_prob,
                                              otsu_threshold(selection.msnet_prob))
            score = metrics["auc"] + metrics["acc"] * len(selection) / (len(selection) + 1e-8)
            history[-1].update(selection_auc=metrics["auc"], selection_accuracy=metrics["acc"],
                               selection_score=score)
            if score > best_score:
                best_score = score
                save_state(best_path, model.state_dict())
        scheduler.step(epoch)
        history[-1]["lr"] = float(optimizer.param_groups[0]["lr"])
        pd.DataFrame(history).to_csv(history_path, index=False)
        save_state(state_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(), "next_epoch": epoch + 1,
                                "global_steps": global_steps, "torch_rng": torch.get_rng_state(),
                                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
        print(f"epoch={epoch + 1}/{epochs} train_bce={history[-1]['train_bce']:.6f}", flush=True)
    if selection_indices is not None:
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    save_state(final_path, model.state_dict())
    return model, final_path


@torch.inference_mode()
def msnet_scores(model, cache, metadata, indices, device, alpha=1.0) -> pd.DataFrame:
    model.eval()
    loader = DataLoader(CachedSubjects(cache, metadata, indices), batch_size=1, shuffle=False, num_workers=0)
    rows = []
    for features, labels, subjects in loader:
        score = float(model(features.to(device), alpha=alpha).item())
        rows.append({"subject_int": int(subjects.item()), "label": int(labels.item()), "msnet_prob": score})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--behavior-features", type=Path, required=True)
    parser.add_argument("--feature-sets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-fold", choices=[f"Set_{i}" for i in range(4)], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ems-checkpoint-selection", action="store_true",
                        help="Use EMS batch size 8 by default and select/reload checkpoints on the middle fold")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("CUDA is required for this run")
    if args.batch_size is None:
        args.batch_size = 8 if args.ems_checkpoint_selection else 4
    train_folds, alpha_fold, evaluation_fold = fold_roles(args.eval_fold)
    folder = args.output_root / evaluation_fold
    folder.mkdir(parents=True, exist_ok=True)
    config = {"eval_fold": evaluation_fold, "alpha_fold": alpha_fold, "train_folds": list(train_folds),
              "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
              "cache_sha256": sha256(args.cache_dir / "features_100.npy"),
              "behavior_sha256": sha256(args.behavior_features), "feature_sets_sha256": sha256(args.feature_sets)}
    if args.ems_checkpoint_selection:
        config.update(checkpoint_rule="middle-fold Otsu accuracy + AUC; reload best before each next epoch",
                      metadata_sha256=sha256(args.cache_dir / "subjects.csv"),
                      train_source_sha256=sha256(Path(__file__)), torch_version=str(torch.__version__))
    config_path = folder / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Existing run_config.json differs; choose a new output-root")
    if (folder / "completed.json").exists():
        print(f"{evaluation_fold} already complete", flush=True)
        return
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    metadata = pd.read_csv(args.cache_dir / "subjects.csv")
    validate_subject_map(metadata)
    cache = np.load(args.cache_dir / "features_100.npy", mmap_mode="r")
    if cache.shape != (160, 1056, 100, 14) or cache.dtype != np.float32:
        raise ValueError(f"Unexpected cache: {cache.shape}, {cache.dtype}")
    behavior = pd.read_csv(args.behavior_features)
    columns = json.loads(args.feature_sets.read_text(encoding="utf-8"))["pupil_scanpath"]
    if len(columns) != 173 or behavior.subject_int.nunique() != 160:
        raise ValueError("Expected 173 P+S columns and 160 behavioral subjects")
    behavior = metadata[["subject_int", "set_name", "label"]].merge(
        behavior.drop(columns=["set_name", "label", "subject_id"], errors="ignore"),
        on="subject_int", how="left", validate="one_to_one")
    if behavior[columns].isna().all(axis=1).any():
        raise ValueError("Missing behavioral subject rows")
    train_indices = np.flatnonzero(metadata.set_name.isin(train_folds))
    alpha_indices = np.flatnonzero(metadata.set_name == alpha_fold)
    eval_indices = np.flatnonzero(metadata.set_name == evaluation_fold)
    if (len(train_indices), len(alpha_indices), len(eval_indices)) != (80, 40, 40):
        raise ValueError("Expected split sizes 80/40/40")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = train_msnet(cache, metadata, train_indices, folder, args.epochs, args.batch_size,
                                    args.seed, device, alpha_indices if args.ems_checkpoint_selection else None)
    model_alpha = msnet_scores(model, cache, metadata, alpha_indices, device)
    model_eval = msnet_scores(model, cache, metadata, eval_indices, device)
    behavior_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03,
                 max_leaf_nodes=15, l2_regularization=0.01, random_state=args.seed)),
    ])
    behavior_model.fit(behavior.iloc[train_indices][columns], behavior.iloc[train_indices].label.astype(int))
    def combine(indices, scores):
        part = metadata.iloc[indices][["subject_int", "subject_id", "set_name", "label"]].copy()
        part["behavior_prob"] = behavior_model.predict_proba(behavior.iloc[indices][columns])[:, 1]
        return part.merge(scores[["subject_int", "msnet_prob"]], on="subject_int", validate="one_to_one")
    calibration = combine(alpha_indices, model_alpha)
    evaluated = combine(eval_indices, model_eval)
    choice = select_alpha(calibration.label, calibration.msnet_prob, calibration.behavior_prob)
    alpha_candidates(calibration.label, calibration.msnet_prob, calibration.behavior_prob).to_csv(
        folder / "alpha_calibration_curve.csv", index=False)
    for part in (calibration, evaluated):
        part["alpha_msnet"] = choice["alpha_msnet"]
        part["fused_prob"] = (choice["alpha_msnet"] * part.msnet_prob +
                              (1 - choice["alpha_msnet"]) * part.behavior_prob)
        part["threshold"] = choice["threshold"]
        part["pred"] = (part.fused_prob >= choice["threshold"]).astype(int)
    calibration.to_csv(folder / "calibration_predictions.csv", index=False)
    evaluated.to_csv(folder / "evaluation_predictions.csv", index=False)
    result = {"fold": evaluation_fold, "alpha_fold": alpha_fold, "train_folds": train_folds,
              "alpha_msnet": choice["alpha_msnet"], "calibration_auc": choice["auc"],
              **fixed_threshold_metrics(evaluated.label, evaluated.fused_prob, choice["threshold"])}
    pd.DataFrame([result]).to_csv(folder / "evaluation_metrics.csv", index=False)
    (folder / "completed.json").write_text(json.dumps({"config": config, "result": result,
        "checkpoint_sha256": sha256(checkpoint),
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()}, indent=2), encoding="utf-8")
    print(f"{evaluation_fold}: eval AUC={result['auc']:.6f}, alpha={choice['alpha_msnet']:.2f}", flush=True)


if __name__ == "__main__":
    main()
