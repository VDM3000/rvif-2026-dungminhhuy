from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    CATEGORY_ORDER,
    FOLDS,
    REPO_DIR,
    cache_dir,
    category_dir,
    compute_metrics,
    load_manifest,
    set_seed,
    write_json,
)


sys.path.insert(0, str(REPO_DIR))
from MSNet import Model


class CategoryDataset(Dataset):
    def __init__(self, features: np.ndarray, metadata: pd.DataFrame, indices: np.ndarray):
        self.features = features
        self.metadata = metadata.iloc[indices].reset_index(drop=True)
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        feature = self.features[source_index]
        if not feature.flags.writeable:
            feature = np.array(feature, copy=True)
        return (
            torch.from_numpy(feature),
            torch.tensor(float(self.metadata.iloc[index]["label"]), dtype=torch.float32),
            int(self.metadata.iloc[index]["subject_int"]),
        )


def make_loaders(
    features: np.ndarray,
    metadata: pd.DataFrame,
    fold: str,
    batch_size: int,
    seed: int,
    epoch: int,
) -> tuple[DataLoader, DataLoader]:
    train_indices = np.flatnonzero(metadata["set_name"].values != fold)
    val_indices = np.flatnonzero(metadata["set_name"].values == fold)
    generator = torch.Generator().manual_seed(seed + epoch)
    train_loader = DataLoader(
        CategoryDataset(features, metadata, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        CategoryDataset(features, metadata, val_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, alpha: float):
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    subject_ids: list[int] = []
    for features, target, subjects in loader:
        features = features.to(device, non_blocking=True)
        probability = model(features, alpha=alpha).flatten().cpu().numpy()
        labels.extend(target.numpy().astype(int).tolist())
        probabilities.extend(probability.astype(float).tolist())
        subject_ids.extend(subjects.numpy().astype(int).tolist())
    return np.asarray(labels), np.asarray(probabilities), np.asarray(subject_ids)


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def train_one(
    category: str,
    fold: str,
    epochs: int,
    batch_size: int,
    seed: int,
    preload: bool,
    run_name: str,
    force: bool,
) -> None:
    manifest = load_manifest(category)
    stimulus_count = int(manifest["image_count"])
    folder = category_dir(category) / ("msnet" if run_name == "main" else f"msnet_{run_name}") / fold
    folder.mkdir(parents=True, exist_ok=True)
    completion_path = folder / "completed.json"
    if completion_path.exists() and not force:
        complete = json.loads(completion_path.read_text(encoding="utf-8"))
        if int(complete.get("epochs", -1)) == epochs:
            print(f"[{category}/{fold}] already complete; skipping")
            return

    feature_path = cache_dir(category) / "features.npy"
    metadata = pd.read_csv(cache_dir(category) / "subjects.csv")
    print(f"[{category}/{fold}] loading {feature_path.name}; preload={preload}")
    started = time.perf_counter()
    features = np.load(feature_path, mmap_mode=None if preload else "r")
    print(
        f"[{category}/{fold}] shape={features.shape}, loaded={time.perf_counter()-started:.1f}s, "
        f"available_ram={psutil.virtual_memory().available / 1024**3:.1f}GB"
    )

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(in_c=1056, stimulus_count=stimulus_count).to(device)
    optimizer = create_optimizer_v2(
        model,
        opt="adamw",
        lr=5e-4,
        weight_decay=0.05,
        eps=1e-8,
    )
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=epochs,
        min_lr=1e-5,
        warmup_lr=1e-6,
        warmup_epochs=min(5, max(0, epochs - 1)),
        cooldown_epochs=min(10, max(0, epochs // 5)),
    )
    criterion = nn.BCELoss()
    best_path = folder / "best_model.pt"
    state_path = folder / "training_state.pt"
    history_path = folder / "training_history.csv"
    best_score = -np.inf
    best_epoch = -1
    global_steps = -1
    start_epoch = 0
    history: list[dict] = []

    if state_path.exists() and not force:
        state = torch.load(state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        global_steps = int(state["global_steps"])
        start_epoch = int(state["next_epoch"])
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        print(f"[{category}/{fold}] resuming at epoch {start_epoch}")

    run_started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        if epoch > 0 and best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        train_loader, val_loader = make_loaders(
            features, metadata, fold, batch_size, seed, epoch
        )
        model.train()
        losses: list[float] = []
        # Epoch summaries below are sufficient for unattended multi-fold runs and
        # keep Windows terminal logs compact. Enable the bar only in an interactive TTY.
        progress = tqdm(
            train_loader,
            desc=f"{category}/{fold} epoch {epoch+1}/{epochs}",
            leave=False,
            disable=not sys.stdout.isatty(),
        )
        for batch_features, target, _ in progress:
            global_steps += 1
            batch_features = batch_features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).unsqueeze(-1)
            optimizer.zero_grad(set_to_none=True)
            probability = model(batch_features, alpha=min(1.0, global_steps / 50.0))
            loss = criterion(probability, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{np.mean(losses[-5:]):.4f}")

        labels, probabilities, _ = predict(
            model, val_loader, device, alpha=min(1.0, global_steps / 50.0)
        )
        val_auc = float(roc_auc_score(labels, probabilities))
        threshold_metrics = compute_metrics(labels, probabilities)
        score = val_auc + threshold_metrics["acc"]
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_auc": val_auc,
            "val_acc": threshold_metrics["acc"],
            "val_f1": threshold_metrics["f1_score"],
            "selection_score": score,
            "best_score": best_score,
            "improved": improved,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - run_started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"[{category}/{fold}] epoch={epoch+1} loss={row['train_loss']:.4f} "
            f"auc={val_auc:.4f} acc={row['val_acc']:.4f} best_epoch={best_epoch+1}"
        )
        scheduler.step(epoch + 1)
        atomic_torch_save(
            {
                "next_epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "global_steps": global_steps,
            },
            state_path,
        )

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _, val_loader = make_loaders(features, metadata, fold, batch_size, seed, epochs)
    labels, probabilities, subject_ids = predict(model, val_loader, device, alpha=1.0)
    prediction_frame = pd.DataFrame(
        {
            "subject_int": subject_ids,
            "subject_id": [f"{value:03d}" for value in subject_ids],
            "set_name": fold,
            "label": labels,
            "msnet_prob": probabilities,
        }
    ).sort_values("subject_int")
    prediction_frame.to_csv(folder / "prob_val.csv", index=False)
    final_metrics = compute_metrics(labels, probabilities)
    write_json(
        folder / "fold_metrics.json",
        {"category": category, "set_name": fold, "best_epoch": best_epoch, **final_metrics},
    )
    write_json(
        completion_path,
        {
            "category": category,
            "set_name": fold,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "stimulus_count": stimulus_count,
            "best_epoch": best_epoch,
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - run_started,
        },
    )
    print(f"[{category}/{fold}] COMPLETE metrics={final_metrics}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["all", *CATEGORY_ORDER], required=True)
    parser.add_argument("--fold", choices=["all", *FOLDS], default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    categories = CATEGORY_ORDER if args.category == "all" else (args.category,)
    folds = FOLDS if args.fold == "all" else (args.fold,)
    for category in categories:
        for fold in folds:
            train_one(
                category,
                fold,
                args.epochs,
                args.batch_size,
                args.seed,
                args.preload,
                args.run_name,
                args.force,
            )


if __name__ == "__main__":
    main()
