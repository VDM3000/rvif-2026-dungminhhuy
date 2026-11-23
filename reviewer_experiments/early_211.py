"""Frozen-MSNet early fusion on the same 2/1/1 folds as late fusion."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .protocol_211 import fold_roles, otsu_threshold, validate_subject_map
from .seed_alpha import sha256
from .train_211 import CachedSubjects, fixed_threshold_metrics, seed_everything


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "EMS_repo"))
from MSNet import Model


def extract_embedding(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Capture normalized MSNet features immediately before its 56-input head."""
    captured = []
    hook = model.head.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach().flatten(1)))
    try:
        with torch.inference_mode():
            model(features, alpha=1.0)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].shape[1] != 56:
        raise RuntimeError(f"Expected one 56-dimensional MSNet embedding, got {[x.shape for x in captured]}")
    return captured[0]


HEAD_DESCRIPTIONS = {
    "mlp": "Linear(229,32), ReLU, Dropout(0.2), Linear(32,1)",
    "linear": "Linear(229,1)",
}


def make_head(head_type="mlp") -> nn.Sequential:
    if head_type == "linear":
        return nn.Sequential(nn.Linear(229, 1))
    if head_type != "mlp":
        raise ValueError(f"Unknown head type: {head_type}")
    return nn.Sequential(nn.Linear(229, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))


def all_embeddings(model, cache, device) -> np.ndarray:
    model.eval()
    values = []
    for index in range(len(cache)):
        x = torch.from_numpy(np.array(cache[index], copy=True)).unsqueeze(0).to(device)
        values.append(extract_embedding(model, x).cpu().numpy()[0])
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (160, 56) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid MSNet embedding matrix: {result.shape}")
    return result


def fit_head(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, calibration_idx: np.ndarray,
             seed: int, device, output: Path, head_type="mlp") -> tuple[nn.Module, int, float]:
    seed_everything(seed)
    head = make_head(head_type).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    x_device = torch.from_numpy(x).to(device)
    y_device = torch.from_numpy(y.astype(np.float32)).to(device)
    best_auc = -1.0
    best_epoch = -1
    waiting = 0
    history = []
    for epoch in range(1, 101):
        generator = torch.Generator().manual_seed(seed + epoch)
        train_data = TensorDataset(x_device[train_idx].cpu(), y_device[train_idx].cpu())
        loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=generator, num_workers=0)
        head.train()
        losses = []
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        head.eval()
        with torch.inference_mode():
            calibration_scores = torch.sigmoid(head(x_device[calibration_idx])).flatten().cpu().numpy()
        auc = float(roc_auc_score(y[calibration_idx], calibration_scores))
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses)), "calibration_auc": auc})
        if auc > best_auc + 1e-12:
            best_auc = auc
            best_epoch = epoch
            waiting = 0
            torch.save(head.state_dict(), output / "head.pt")
        else:
            waiting += 1
        if waiting >= 10:
            break
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    head.load_state_dict(torch.load(output / "head.pt", map_location=device, weights_only=True))
    head.eval()
    return head, best_epoch, best_auc


@torch.inference_mode()
def head_scores(head, x: np.ndarray, indices: np.ndarray, device) -> np.ndarray:
    return torch.sigmoid(head(torch.from_numpy(x[indices]).to(device))).flatten().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--behavior-features", type=Path, required=True)
    parser.add_argument("--feature-sets", type=Path, required=True)
    parser.add_argument("--late-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-fold", choices=[f"Set_{i}" for i in range(4)], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-type", choices=HEAD_DESCRIPTIONS, default="mlp")
    args = parser.parse_args()
    train_folds, alpha_fold, eval_fold = fold_roles(args.eval_fold)
    late_folder = args.late_root / eval_fold
    late_complete = json.loads((late_folder / "completed.json").read_text(encoding="utf-8"))
    late_config = late_complete["config"]
    if (late_config["eval_fold"] != eval_fold or late_config["alpha_fold"] != alpha_fold or
            tuple(late_config["train_folds"]) != train_folds or late_config["epochs"] != 50):
        raise ValueError("Late-fusion checkpoint does not follow the expected 2/1/1 protocol")
    checkpoint = late_folder / "final_model.pt"
    if sha256(checkpoint) != late_complete["checkpoint_sha256"]:
        raise ValueError("MSNet checkpoint hash differs from the completed late-fusion run")
    folder = args.output_root / eval_fold
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "completed.json").exists():
        saved = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
        if saved.get("head_type", "mlp") != args.head_type or saved["msnet_checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError("Completed run used a different head or MSNet checkpoint")
        print(f"{eval_fold} already complete", flush=True)
        return
    metadata = pd.read_csv(args.cache_dir / "subjects.csv")
    validate_subject_map(metadata)
    cache = np.load(args.cache_dir / "features_100.npy", mmap_mode="r")
    if cache.shape != (160, 1056, 100, 14):
        raise ValueError(f"Wrong cache shape: {cache.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(in_c=1056).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    embeddings = all_embeddings(model, cache, device)
    embedding_frame = pd.DataFrame(embeddings, columns=[f"msnet_feature_{i:02d}" for i in range(56)])
    embedding_frame.insert(0, "subject_int", metadata.subject_int)
    embedding_frame.to_csv(folder / "msnet_embeddings.csv", index=False)
    behavior = pd.read_csv(args.behavior_features)
    columns = json.loads(args.feature_sets.read_text(encoding="utf-8"))["pupil_scanpath"]
    if len(columns) != 173:
        raise ValueError(f"Expected 173 behavioral features, got {len(columns)}")
    aligned = metadata[["subject_int"]].merge(behavior[["subject_int", *columns]],
                                               on="subject_int", how="left", validate="one_to_one")
    if aligned[columns].isna().all(axis=1).any():
        raise ValueError("Missing behavioral subject rows")
    train_idx = np.flatnonzero(metadata.set_name.isin(train_folds))
    calibration_idx = np.flatnonzero(metadata.set_name == alpha_fold)
    eval_idx = np.flatnonzero(metadata.set_name == eval_fold)
    if (len(train_idx), len(calibration_idx), len(eval_idx)) != (80, 40, 40):
        raise ValueError("Expected 80/40/40 subjects")
    imputer = SimpleImputer(strategy="median").fit(aligned.iloc[train_idx][columns])
    imputed = imputer.transform(aligned[columns])
    scaler = StandardScaler().fit(imputed[train_idx])
    behavior_scaled = scaler.transform(imputed).astype(np.float32)
    combined = np.concatenate((embeddings, behavior_scaled), axis=1)
    if combined.shape != (160, 229) or not np.isfinite(combined).all():
        raise ValueError(f"Invalid combined vector: {combined.shape}")
    joblib.dump({"imputer": imputer, "scaler": scaler, "behavior_columns": columns}, folder / "preprocessing.joblib")
    labels = metadata.label.to_numpy(dtype=int)
    head, best_epoch, calibration_auc = fit_head(combined, labels, train_idx, calibration_idx,
                                                 args.seed, device, folder, args.head_type)
    calibration_scores = head_scores(head, combined, calibration_idx, device)
    threshold = otsu_threshold(calibration_scores)
    evaluation_scores = head_scores(head, combined, eval_idx, device)
    calibration = metadata.iloc[calibration_idx][["subject_int", "subject_id", "set_name", "label"]].copy()
    calibration["early_prob"] = calibration_scores
    calibration["threshold"] = threshold
    calibration["pred"] = (calibration_scores >= threshold).astype(int)
    evaluation = metadata.iloc[eval_idx][["subject_int", "subject_id", "set_name", "label"]].copy()
    evaluation["early_prob"] = evaluation_scores
    evaluation["threshold"] = threshold
    evaluation["pred"] = (evaluation_scores >= threshold).astype(int)
    calibration.to_csv(folder / "calibration_predictions.csv", index=False)
    evaluation.to_csv(folder / "evaluation_predictions.csv", index=False)
    result = {"fold": eval_fold, "alpha_fold": alpha_fold, "train_folds": train_folds,
              "best_epoch": best_epoch, "calibration_auc": calibration_auc,
              **fixed_threshold_metrics(evaluation.label, evaluation.early_prob, threshold)}
    pd.DataFrame([result]).to_csv(folder / "evaluation_metrics.csv", index=False)
    (folder / "completed.json").write_text(json.dumps({
        "result": result, "msnet_checkpoint_sha256": sha256(checkpoint), "head_sha256": sha256(folder / "head.pt"),
        "behavior_sha256": sha256(args.behavior_features), "feature_sets_sha256": sha256(args.feature_sets),
        "head_type": args.head_type,
        "architecture": f"frozen MSNet encoder; 56+173=229; {HEAD_DESCRIPTIONS[args.head_type]}",
        "optimizer": "AdamW lr=0.001; BCEWithLogitsLoss; max_epoch=100; calibration AUC patience=10",
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()}, indent=2), encoding="utf-8")
    print(f"{eval_fold}: eval AUC={result['auc']:.6f}, best_epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
