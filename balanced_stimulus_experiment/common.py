from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORY_CACHE_ROOT = PROJECT_ROOT / "work" / "category_experiments" / "cache"
WORK_ROOT = PROJECT_ROOT / "work" / "balanced_stimulus_experiment"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "balanced_stimulus_experiment"
MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
CACHE_PATH = WORK_ROOT / "features_60.npy"
SUBJECTS_PATH = WORK_ROOT / "subjects.csv"
MASTER_MANIFEST_PATH = WORK_ROOT / "master_manifest.json"
STIMULUS_COUNTS = (20, 40, 60)
FOLDS = ("Set_0", "Set_1", "Set_2", "Set_3")
CATEGORY_ORDER = ("manipulated", "natural", "social", "synthetic")
CATEGORY_NAMES = {
    "manipulated": "Manipulated Images",
    "natural": "Natural Scenes",
    "social": "Social Scenes",
    "synthetic": "Synthetic Images",
}
SEED = 42
ALPHA_MSNET = 0.8
BEHAVIOR_WEIGHT = 0.2


def condition_name(stimulus_count: int) -> str:
    if stimulus_count not in STIMULUS_COUNTS:
        raise ValueError(stimulus_count)
    return f"subset_{stimulus_count}"


def result_dir(stimulus_count: int) -> Path:
    path = OUTPUT_ROOT / f"results_{stimulus_count}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(stimulus_count: int) -> Path:
    return MANIFEST_ROOT / f"stimulus_{stimulus_count}.csv"


def load_manifest(stimulus_count: int) -> pd.DataFrame:
    path = manifest_path(stimulus_count)
    if not path.exists():
        raise FileNotFoundError(f"Run prepare.py first; missing {path}")
    frame = pd.read_csv(path)
    expected = {"IMAGE", "CATEGORY"}
    if set(frame.columns) != expected or len(frame) != stimulus_count:
        raise RuntimeError(f"Invalid manifest {path}")
    frame["image_stem"] = frame["IMAGE"].map(lambda value: Path(str(value)).stem)
    return frame


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def master_cache(stimulus_count: int, preload: bool = True) -> np.ndarray:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Run prepare.py first; missing {CACHE_PATH}")
    features = np.load(CACHE_PATH, mmap_mode=None if preload else "r")
    expected = (160, 1056, 60, 14)
    if features.shape != expected:
        raise RuntimeError(f"Unexpected master cache shape: {features.shape}")
    return features[:, :, :stimulus_count, :]
