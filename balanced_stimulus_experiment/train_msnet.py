from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import roc_auc_score
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    FOLDS,
    PROJECT_ROOT,
    STIMULUS_COUNTS,
    SUBJECTS_PATH,
    master_cache,
    result_dir,
    write_json,
)


sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "EMS_repo"))
from MSNet import Model
from category_experiments.common import compute_metrics, set_seed


class BalancedDataset(Dataset):
    def __init__(self, features: np.ndarray, metadata: pd.DataFrame, indices: np.ndarray):
        self.features = features
        self.metadata = metadata.iloc[indices].reset_index(drop=True)
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        feature = self.features[source_index]
        if not feature.flags.writeable or not feature.flags.c_contiguous:
            feature = np.ascontiguousarray(feature)
        return (
            torch.from_numpy(feature),
            torch.tensor(float(self.metadata.iloc[index]["label"]), dtype=torch.float32),
            int(self.metadata.iloc[index]["subject_int"]),
        )


def make_loaders(features, metadata, fold, batch_size, seed, epoch):
    train_indices = np.flatnonzero(metadata["set_name"].values != fold)
    val_indices = np.flatnonzero(metadata["set_name"].values == fold)
    generator = torch.Generator().manual_seed(seed + epoch)
    train = DataLoader(
        BalancedDataset(features, metadata, train_indices),
        batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available(), generator=generator,
    )
    validation = DataLoader(
        BalancedDataset(features, metadata, val_indices),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train, validation


@torch.inference_mode()
def predict(model, loader, device, alpha):
    model.eval()
    labels, probabilities, subject_ids = [], [], []
    for features, target, subjects in loader:
        probability = model(features.to(device, non_blocking=True), alpha=alpha)
        labels.extend(target.numpy().astype(int).tolist())
        probabilities.extend(probability.flatten().cpu().numpy().astype(float).tolist())
        subject_ids.extend(subjects.numpy().astype(int).tolist())
    return np.asarray(labels), np.asarray(probabilities), np.asarray(subject_ids)


def atomic_torch_save(value: dict, path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def train_one(stimulus_count, fold, epochs, batch_size, seed, preload, force):
    folder = result_dir(stimulus_count) / "msnet" / fold
    folder.mkdir(parents=True, exist_ok=True)
    completion_path = folder / "completed.json"
    if completion_path.exists() and not force:
        complete = json.loads(completion_path.read_text(encoding="utf-8"))
        if int(complete.get("epochs", -1)) == epochs and int(
            complete.get("stimulus_count", -1)
        ) == stimulus_count:
            print(f"[{stimulus_count}/{fold}] already complete; skipping")
            return

    metadata = pd.read_csv(SUBJECTS_PATH)
    print(f"[{stimulus_count}/{fold}] loading master prefix; preload={preload}")
    started = time.perf_counter()
    features = master_cache(stimulus_count, preload=preload)
    if preload:
        features = np.ascontiguousarray(features)
    print(
        f"[{stimulus_count}/{fold}] shape={features.shape}, loaded={time.perf_counter()-started:.1f}s, "
        f"available_ram={psutil.virtual_memory().available / 1024**3:.1f}GB"
    )

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(in_c=1056, stimulus_count=stimulus_count).to(device)
    optimizer = create_optimizer_v2(
        model, opt="adamw", lr=5e-4, weight_decay=0.05, eps=1e-8
    )
    scheduler, _ = create_scheduler_v2(
        optimizer, sched="cosine", num_epochs=epochs, min_lr=1e-5,
        warmup_lr=1e-6, warmup_epochs=min(5, max(0, epochs - 1)),
        cooldown_epochs=min(10, max(0, epochs // 5)),
    )
    criterion = nn.BCELoss()
    best_path = folder / "best_model.pt"
    state_path = folder / "training_state.pt"
    history_path = folder / "training_history.csv"
    best_score, best_epoch, global_steps, start_epoch = -np.inf, -1, -1, 0
    history: list[dict] = []

    if state_path.exists() and not force:
        state = torch.load(state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        global_steps = int(state["global_steps"])
        start_epoch = int(state["next_epoch"])
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        print(f"[{stimulus_count}/{fold}] resuming at epoch {start_epoch}")

    run_started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        if epoch > 0 and best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        train_loader, val_loader = make_loaders(
            features, metadata, fold, batch_size, seed, epoch
        )
        model.train()
        losses = []
        progress = tqdm(
            train_loader, desc=f"{stimulus_count}/{fold} epoch {epoch+1}/{epochs}",
            leave=False, disable=not sys.stdout.isatty(),
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
        labels, probabilities, _ = predict(
            model, val_loader, device, alpha=min(1.0, global_steps / 50.0)
        )
        val_auc = float(roc_auc_score(labels, probabilities))
        threshold_metrics = compute_metrics(labels, probabilities)
        score = val_auc + threshold_metrics["acc"]
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), best_path)
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": val_auc,
            "val_acc": threshold_metrics["acc"], "val_f1": threshold_metrics["f1_score"],
            "selection_score": score, "best_score": best_score, "improved": improved,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - run_started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"[{stimulus_count}/{fold}] epoch={epoch+1} loss={row['train_loss']:.4f} "
            f"auc={val_auc:.4f} acc={row['val_acc']:.4f} best_epoch={best_epoch+1}"
        )
        scheduler.step(epoch + 1)
        atomic_torch_save(
            {"next_epoch": epoch + 1, "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "best_score": best_score,
             "best_epoch": best_epoch, "global_steps": global_steps}, state_path,
        )

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _, val_loader = make_loaders(features, metadata, fold, batch_size, seed, epochs)
    labels, probabilities, subject_ids = predict(model, val_loader, device, alpha=1.0)
    predictions = pd.DataFrame({
        "subject_int": subject_ids,
        "subject_id": [f"{value:03d}" for value in subject_ids],
        "set_name": fold, "label": labels, "msnet_prob": probabilities,
    }).sort_values("subject_int")
    predictions.to_csv(folder / "prob_val.csv", index=False)
    final_metrics = compute_metrics(labels, probabilities)
    write_json(folder / "fold_metrics.json", {
        "stimulus_count": stimulus_count, "set_name": fold,
        "best_epoch": best_epoch + 1, **final_metrics,
    })
    write_json(completion_path, {
        "stimulus_count": stimulus_count, "set_name": fold, "epochs": epochs,
        "batch_size": batch_size, "seed": seed, "best_epoch": best_epoch + 1,
        "device": str(device), "elapsed_seconds": time.perf_counter() - run_started,
    })
    print(f"[{stimulus_count}/{fold}] COMPLETE metrics={final_metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimulus-count", choices=["all", *map(str, STIMULUS_COUNTS)], default="all")
    parser.add_argument("--fold", choices=["all", *FOLDS], default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    counts = STIMULUS_COUNTS if args.stimulus_count == "all" else (int(args.stimulus_count),)
    folds = FOLDS if args.fold == "all" else (args.fold,)
    for count in counts:
        for fold in folds:
            train_one(count, fold, args.epochs, args.batch_size, args.seed, args.preload, args.force)


if __name__ == "__main__":
    main()
