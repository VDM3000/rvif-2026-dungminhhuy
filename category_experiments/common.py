from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn import metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "EMS"
REPO_DIR = PROJECT_ROOT / "EMS_repo"
FEATURE_DICT_PATH = PROJECT_ROOT / "feature_dict_RINet.npy"
EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "category_experiments"
CACHE_ROOT = PROJECT_ROOT / "work" / "category_experiments" / "cache"
FULL_RESULTS_DIR = PROJECT_ROOT / "improve_behavioral_final_results"
FOLDS = ("Set_0", "Set_1", "Set_2", "Set_3")
CATEGORY_ORDER = (
    "manipulated",
    "natural",
    "social",
    "synthetic",
)
CATEGORY_DIRS = OrderedDict(
    [
        ("manipulated", "Manipulated Images"),
        ("natural", "Natural Scenes"),
        ("social", "Social Scenes"),
        ("synthetic", "Synthetic Images"),
    ]
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_sets_from_disk() -> OrderedDict[str, set[str]]:
    result: OrderedDict[str, set[str]] = OrderedDict()
    for slug, dirname in CATEGORY_DIRS.items():
        folder = DATASET_DIR / "Images" / dirname
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing image category: {folder}")
        result[slug] = {path.stem for path in folder.iterdir() if path.is_file()}
    union = set().union(*result.values())
    if len(union) != 100 or sum(map(len, result.values())) != 100:
        raise RuntimeError("The four category folders must contain 100 unique images")
    return result


def ordered_category_images(feature_keys: list[str]) -> OrderedDict[str, list[str]]:
    disk_sets = image_sets_from_disk()
    result: OrderedDict[str, list[str]] = OrderedDict()
    for slug, names in disk_sets.items():
        result[slug] = [key for key in feature_keys if key in names]
        if set(result[slug]) != names:
            missing = sorted(names - set(result[slug]))
            raise RuntimeError(f"RINet feature dictionary is missing {slug} images: {missing}")
    return result


def subject_metadata() -> pd.DataFrame:
    workbook = pd.read_excel(DATASET_DIR / "Train_Valid.xlsx")
    rows: list[dict] = []
    seen: set[int] = set()
    for fold in FOLDS:
        for value in workbook[fold].dropna().tolist():
            subject_int = int(value)
            if subject_int in seen:
                raise RuntimeError(f"Subject appears in more than one fold: {subject_int}")
            seen.add(subject_int)
            rows.append(
                {
                    "subject_int": subject_int,
                    "subject_id": f"{subject_int:03d}",
                    "set_name": fold,
                    "label": 0 if subject_int < 200 else 1,
                }
            )
    result = pd.DataFrame(rows).sort_values("subject_int").reset_index(drop=True)
    if len(result) != 160:
        raise RuntimeError(f"Expected 160 validation subjects, found {len(result)}")
    return result


def category_dir(category: str) -> Path:
    if category not in CATEGORY_ORDER:
        raise ValueError(f"Unknown category: {category}")
    path = EXPERIMENT_ROOT / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(category: str) -> Path:
    if category not in CATEGORY_ORDER:
        raise ValueError(f"Unknown category: {category}")
    path = CACHE_ROOT / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest(category: str) -> dict:
    path = cache_dir(category) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare_cache.py first; missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def otsu_threshold(y_prob: np.ndarray) -> float:
    scaled = np.expand_dims(np.asarray(y_prob, dtype=float) * 255, axis=-1).astype(np.uint8)
    threshold, _ = cv2.threshold(scaled.copy(), 0, 255, cv2.THRESH_OTSU)
    return float(threshold) / 255.0


def compute_metrics(y_true, y_prob, threshold: float | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if threshold is None:
        threshold = otsu_threshold(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "acc": float(metrics.accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "auc": float(metrics.roc_auc_score(y_true, y_prob)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_foldwise_metrics(
    frame: pd.DataFrame,
    prob_col: str,
    folds: tuple[str, ...] = FOLDS,
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold in folds:
        fold_df = frame[frame["set_name"] == fold]
        if fold_df.empty:
            continue
        rows.append(
            {
                "set_name": fold,
                **compute_metrics(fold_df["label"].values, fold_df[prob_col].values),
            }
        )
    fold_metrics = pd.DataFrame(rows)
    avg = {"set_name": "average"}
    for column in [
        "threshold",
        "acc",
        "sensitivity",
        "specificity",
        "auc",
        "precision",
        "f1_score",
    ]:
        avg[column] = float(fold_metrics[column].mean())
    for column in ["tn", "fp", "fn", "tp"]:
        avg[column] = np.nan
    return pd.concat([fold_metrics, pd.DataFrame([avg])], ignore_index=True)


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
