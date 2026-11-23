"""Early fusion on the historical 120-train/40-validation EMS folds.

The pretrained MSNet encoder is the one used by windows.ipynb. A new 229-input
head is trained on each fold's 120 development subjects. Its epoch is chosen
inside those 120 subjects; the outer validation labels are read only for final
metrics. This is development validation, not an independent test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .early_211 import HEAD_DESCRIPTIONS, all_embeddings, fit_head, head_scores, make_head
from .protocol_211 import FOLDS, load_subject_map, validate_subject_map
from .seed_alpha import fold_metrics, metrics, sha256
from .summarize_211 import audit_prediction_metrics
from .train_211 import seed_everything


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "EMS_repo"))
from MSNet import Model


def fold_roles_31(evaluation_fold: str) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Return (all 120 train folds, 80 inner-train folds, 40 epoch-selection fold)."""
    if evaluation_fold not in FOLDS:
        raise ValueError(f"Unknown fold: {evaluation_fold}")
    selection_fold = FOLDS[(FOLDS.index(evaluation_fold) + 1) % 4]
    train_folds = tuple(f for f in FOLDS if f != evaluation_fold)
    selection_train_folds = tuple(f for f in train_folds if f != selection_fold)
    return train_folds, selection_train_folds, selection_fold


def verify_pretrained_parity(observed: pd.DataFrame, expected: pd.DataFrame,
                             evaluation_fold: str, atol: float = 2e-6) -> float:
    """Reject a checkpoint/cache that fails the historical MSNet validation scores."""
    columns = ["subject_int", "set_name", "label"]
    left = observed.loc[observed.set_name == evaluation_fold].sort_values("subject_int").reset_index(drop=True)
    right = expected.loc[expected.set_name == evaluation_fold].sort_values("subject_int").reset_index(drop=True)
    if len(left) != len(right) or len(left) == 0:
        raise ValueError("Expected matched validation subjects")
    if not left[columns].equals(right[columns]):
        raise ValueError("MSNet subject/fold/label mismatch")
    errors = np.abs(left.msnet_prob.to_numpy(dtype=float) - right.msnet_prob.to_numpy(dtype=float))
    max_error = float(errors.max())
    if max_error > atol:
        raise ValueError(f"MSNet probability mismatch: max error {max_error:.3g} > {atol:.3g}")
    return max_error


def prepare_combined(embeddings: np.ndarray, behavior: pd.DataFrame,
                     columns: list[str], fit_indices: np.ndarray):
    """Fit preprocessing only on named training rows and concatenate 56+173."""
    if embeddings.shape != (160, 56) and embeddings.shape[1:] != (56,):
        raise ValueError(f"Expected 56 MSNet features: {embeddings.shape}")
    if len(columns) != 173 or len(behavior) != len(embeddings):
        raise ValueError("Expected aligned 173-column behavioral rows")
    imputer = SimpleImputer(strategy="median").fit(behavior.iloc[fit_indices][columns])
    imputed = imputer.transform(behavior[columns])
    scaler = StandardScaler().fit(imputed[fit_indices])
    combined = np.concatenate((embeddings, scaler.transform(imputed).astype(np.float32)), axis=1)
    if combined.shape != (len(embeddings), 229) or not np.isfinite(combined).all():
        raise ValueError(f"Invalid 229-column combined data: {combined.shape}")
    return combined, imputer, scaler


def train_final_head(x: np.ndarray, labels: np.ndarray, train_indices: np.ndarray,
                     epochs: int, seed: int, device: torch.device, output: Path, head_type="mlp") -> nn.Module:
    """Refit a fresh head on all 120 train subjects for the selected epoch count."""
    if len(train_indices) != 120 or not 1 <= epochs <= 100:
        raise ValueError("Final head needs 120 subjects and 1..100 epochs")
    seed_everything(seed)
    head = make_head(head_type).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    data = TensorDataset(torch.from_numpy(x[train_indices]),
                         torch.from_numpy(labels[train_indices].astype(np.float32)))
    history = []
    for epoch in range(1, epochs + 1):
        loader = DataLoader(data, batch_size=16, shuffle=True,
                            generator=torch.Generator().manual_seed(seed + epoch), num_workers=0)
        head.train()
        losses = []
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(features), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses))})
    head.eval()
    pd.DataFrame(history).to_csv(output / "final_training_history.csv", index=False)
    torch.save(head.state_dict(), output / "head.pt")
    return head


def run_fold(args, cache, metadata: pd.DataFrame, behavior: pd.DataFrame,
             columns: list[str], historical_msnet: pd.DataFrame, evaluation_fold: str,
             cache_sha256: str) -> None:
    train_folds, selection_train_folds, selection_fold = fold_roles_31(evaluation_fold)
    folder = args.output_root / evaluation_fold
    if (folder / "completed.json").exists():
        saved = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if saved.get("head_type", "mlp") != args.head_type or saved["seed"] != args.seed or saved["checkpoint_sha256"] != sha256(
                args.checkpoint_root / evaluation_fold / "best_model.pt") or saved["cache_sha256"] != cache_sha256:
            raise ValueError(f"Completed {evaluation_fold} used different inputs or seed")
        print(f"{evaluation_fold}: already complete", flush=True)
        return
    if folder.exists():
        raise FileExistsError(f"Incomplete output exists; inspect or use a new output root: {folder}")
    checkpoint = args.checkpoint_root / evaluation_fold / "best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing historical pretrained checkpoint: {checkpoint}")
    folder.mkdir(parents=True)
    selection_folder = folder / "selection"
    selection_folder.mkdir()
    manifest = {
        "created_before_run": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "protocol": "historical 3/1 development validation; 120 train, 40 validation",
        "evaluation_fold": evaluation_fold, "training_folds": train_folds,
        "epoch_selection_fold": selection_fold, "epoch_selection_train_folds": selection_train_folds,
        "seed": args.seed, "checkpoint_sha256": sha256(checkpoint),
        "cache_sha256": cache_sha256,
        "behavior_sha256": sha256(args.behavior_features),
        "feature_sets_sha256": sha256(args.feature_sets),
        "historical_msnet_sha256": sha256(args.historical_msnet),
        "source_sha256": sha256(Path(__file__)),
        "repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                      text=True, check=True).stdout.strip(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__, "sklearn": sklearn.__version__,
        "head_type": args.head_type,
        "head": HEAD_DESCRIPTIONS[args.head_type],
        "epoch_rule": "80 inner train, 40 inner selection; best AUC, first tie, patience 10; refit on 120",
        "threshold_rule": "Otsu from outer validation scores without labels, as in historical notebook",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(in_c=1056).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    model.eval()
    embeddings = all_embeddings(model, cache, device)
    embedding_frame = pd.DataFrame(embeddings, columns=[f"msnet_feature_{i:02d}" for i in range(56)])
    embedding_frame.insert(0, "subject_int", metadata.subject_int)
    embedding_frame.to_csv(folder / "msnet_embeddings.csv", index=False)

    validation_indices = np.flatnonzero(metadata.set_name == evaluation_fold)
    with torch.inference_mode():
        old_scores = model.head(torch.from_numpy(embeddings[validation_indices]).to(device)).flatten().cpu().numpy()
    observed = metadata.iloc[validation_indices][["subject_int", "set_name", "label"]].copy()
    observed["msnet_prob"] = old_scores
    max_parity_error = verify_pretrained_parity(observed, historical_msnet, evaluation_fold)

    train_indices = np.flatnonzero(metadata.set_name.isin(train_folds))
    selection_train_indices = np.flatnonzero(metadata.set_name.isin(selection_train_folds))
    selection_indices = np.flatnonzero(metadata.set_name == selection_fold)
    if (len(train_indices), len(selection_train_indices), len(selection_indices), len(validation_indices)) != (120, 80, 40, 40):
        raise ValueError("Expected 120/80/40/40 subject roles")
    labels = metadata.label.to_numpy(dtype=int)
    selection_x, selection_imputer, selection_scaler = prepare_combined(
        embeddings, behavior, columns, selection_train_indices)
    joblib.dump({"imputer": selection_imputer, "scaler": selection_scaler,
                 "behavior_columns": columns}, selection_folder / "preprocessing.joblib")
    selection_head, selected_epoch, selection_auc = fit_head(
        selection_x, labels, selection_train_indices, selection_indices,
        args.seed, device, selection_folder, args.head_type)
    selection_scores = head_scores(selection_head, selection_x, selection_indices, device)
    selected = metadata.iloc[selection_indices][["subject_int", "set_name", "label"]].copy()
    selected["early_prob"] = selection_scores
    selected.to_csv(selection_folder / "selection_predictions.csv", index=False)

    final_x, final_imputer, final_scaler = prepare_combined(embeddings, behavior, columns, train_indices)
    joblib.dump({"imputer": final_imputer, "scaler": final_scaler,
                 "behavior_columns": columns}, folder / "preprocessing.joblib")
    final_head = train_final_head(final_x, labels, train_indices, selected_epoch, args.seed, device, folder, args.head_type)
    validation_scores = head_scores(final_head, final_x, validation_indices, device)
    result = metrics(labels[validation_indices], validation_scores)
    evaluated = metadata.iloc[validation_indices][["subject_int", "subject_id", "set_name", "label"]].copy()
    evaluated["msnet_prob"] = old_scores
    evaluated["early_prob"] = validation_scores
    evaluated["threshold"] = result["threshold"]
    evaluated["pred"] = (validation_scores >= result["threshold"]).astype(int)
    evaluated.to_csv(folder / "validation_predictions.csv", index=False)
    detail = {"evaluation_fold": evaluation_fold, "training_folds": train_folds,
              "epoch_selection_fold": selection_fold, "selected_epoch": selected_epoch,
              "selection_auc": selection_auc, "msnet_parity_max_error": max_parity_error,
              **result}
    pd.DataFrame([detail]).to_csv(folder / "validation_metrics.csv", index=False)
    (folder / "completed.json").write_text(json.dumps({
        "result": detail, "manifest_sha256": sha256(folder / "manifest.json"),
        "head_sha256": sha256(folder / "head.pt"),
        "completed_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
    }, indent=2, default=str), encoding="utf-8")
    print(f"{evaluation_fold}: AUC={result['auc']:.6f}; selected_epoch={selected_epoch}; "
          f"MSNet parity max_error={max_parity_error:.3g}", flush=True)


def summarize(args, expected: pd.DataFrame) -> None:
    frames, rows = [], []
    source = pd.read_csv(args.behavior_features)
    columns = json.loads(args.feature_sets.read_text(encoding="utf-8"))["pupil_scanpath"]
    behavior = expected[["subject_int"]].merge(source[["subject_int", *columns]],
                                                on="subject_int", validate="one_to_one")
    for fold in FOLDS:
        folder = args.output_root / fold
        completion = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
        detail = completion["result"]
        part = pd.read_csv(folder / "validation_predictions.csv")
        if len(part) != 40 or set(part.set_name) != {fold}:
            raise ValueError(f"Wrong validation subjects for {fold}")
        if sha256(folder / "head.pt") != completion["head_sha256"]:
            raise ValueError(f"Saved head hash mismatch in {fold}")
        audit_prediction_metrics(part, detail, "early_prob")
        if not np.isclose(metrics(part.label, part.early_prob)["threshold"], detail["threshold"], atol=1e-12):
            raise ValueError(f"Saved Otsu threshold mismatch in {fold}")
        embedding_frame = pd.read_csv(folder / "msnet_embeddings.csv")
        aligned = expected[["subject_int"]].merge(embedding_frame, on="subject_int", validate="one_to_one")
        embeddings = aligned.drop(columns="subject_int").to_numpy(dtype=np.float32)
        prep = joblib.load(folder / "preprocessing.joblib")
        if prep["behavior_columns"] != columns:
            raise ValueError(f"Behavioral column order mismatch in {fold}")
        scaled = prep["scaler"].transform(prep["imputer"].transform(behavior[columns])).astype(np.float32)
        combined_x = np.concatenate((embeddings, scaled), axis=1)
        head = make_head(args.head_type)
        head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu", weights_only=True))
        head.eval()
        indices = np.flatnonzero(expected.set_name == fold)
        score_map = dict(zip(expected.iloc[indices].subject_int,
                             head_scores(head, combined_x, indices, torch.device("cpu"))))
        if not np.allclose(part.subject_int.map(score_map), part.early_prob, rtol=0, atol=2e-6):
            raise ValueError(f"Saved head does not reproduce probabilities in {fold}")
        frames.append(part)
        rows.append({"method": "Early Fusion 3/1", "fold": fold, **detail})
    combined = pd.concat(frames, ignore_index=True)
    if combined.subject_int.nunique() != 160 or len(combined) != 160:
        raise ValueError("Expected exactly 160 unique validation subjects")
    left = combined[["subject_int", "set_name", "label"]].sort_values("subject_int").reset_index(drop=True)
    right = expected[["subject_int", "set_name", "label"]].sort_values("subject_int").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Validation subject/fold/label mismatch")
    combined.to_csv(args.output_root / "all_validation_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_root / "fold_metrics.csv", index=False)

    baseline = pd.read_csv(args.historical_msnet)
    late = pd.read_csv(args.historical_late)
    metrics_to_report = ("acc", "sensitivity", "specificity", "auc", "precision", "f1_score")
    comparison = []
    for name, frame, column in (("MSNet historical 3/1", baseline, "msnet_prob"),
                                ("Late Fusion historical 3/1", late, "fused_prob"),
                                ("Early Fusion 3/1", combined, "early_prob")):
        foldwise = fold_metrics(frame, column)
        summary = foldwise.loc[foldwise.set_name == "average"].iloc[0]
        comparison.append({"method": name, "protocol": "historical 3/1 development validation",
                           **{key: float(summary[key]) for key in metrics_to_report}})
    pd.DataFrame(comparison).to_csv(args.output_root / "comparison_with_historical.csv", index=False)
    (args.output_root / "interpretation.md").write_text(
        "# Early Fusion 3/1 comparison\n\n"
        "The four predefined EMS validation folds contain 40 people each. MSNet uses the "
        "historical pretrained checkpoint for each fold; the new early-fusion head is refitted "
        "on the other 120 people after choosing its epoch inside those 120. "
        "The validation-score Otsu rule matches the historical notebook and does not use labels. "
        "The historical late-fusion configuration was selected using these same development folds, "
        "so this comparison is descriptive and is not independent validation. "
        "The separate 2/1/1 experiment has different training-set sizes and threshold rules.\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    runs = ROOT / "outputs/reviewer_experiments"
    baseline = runs / "baseline_2026-11-22_seed42/improve_behavioral"
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "work/reviewer_experiments/cache")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "model_parameters")
    parser.add_argument("--behavior-features", type=Path, default=baseline / "subject_behavior_features.csv")
    parser.add_argument("--feature-sets", type=Path, default=baseline / "feature_sets_clean.json")
    parser.add_argument("--historical-msnet", type=Path, default=baseline / "msnet_prob_val.csv")
    parser.add_argument("--historical-late", type=Path, default=baseline / "best_fusion_prob_val.csv")
    parser.add_argument("--output-root", type=Path, default=runs / "early_31_final_2026-11-23")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-fold", choices=[*FOLDS, "all"], default="all")
    parser.add_argument("--head-type", choices=HEAD_DESCRIPTIONS, default="mlp")
    args = parser.parse_args()
    expected = load_subject_map(ROOT / "EMS/Train_Valid.xlsx")
    metadata = pd.read_csv(args.cache_dir / "subjects.csv")
    validate_subject_map(metadata)
    if not metadata[["subject_int", "set_name", "label"]].equals(expected[["subject_int", "set_name", "label"]]):
        raise ValueError("Cached subjects differ from EMS workbook")
    cache = np.load(args.cache_dir / "features_100.npy", mmap_mode="r")
    if cache.shape != (160, 1056, 100, 14) or cache.dtype != np.float32:
        raise ValueError(f"Wrong cached MSNet input: {cache.shape}, {cache.dtype}")
    cache_hash = sha256(args.cache_dir / "features_100.npy")
    features = pd.read_csv(args.behavior_features)
    columns = json.loads(args.feature_sets.read_text(encoding="utf-8"))["pupil_scanpath"]
    if len(columns) != 173 or len(set(columns)) != 173 or features.subject_int.nunique() != 160:
        raise ValueError("Expected 160 unique subjects and 173 P+S features")
    behavior = metadata[["subject_int"]].merge(features[["subject_int", *columns]],
                                                on="subject_int", how="left", validate="one_to_one")
    if behavior[columns].isna().all(axis=1).any():
        raise ValueError("Missing behavioral subject rows")
    historical_msnet = pd.read_csv(args.historical_msnet)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS if args.eval_fold == "all" else (args.eval_fold,):
        run_fold(args, cache, metadata, behavior, columns, historical_msnet, fold, cache_hash)
    if all((args.output_root / fold / "completed.json").exists() for fold in FOLDS):
        summarize(args, expected)
        print(f"Wrote summary to {args.output_root / 'comparison_with_historical.csv'}", flush=True)


if __name__ == "__main__":
    main()
