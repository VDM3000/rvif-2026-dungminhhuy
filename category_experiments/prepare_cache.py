from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    CATEGORY_ORDER,
    FEATURE_DICT_PATH,
    cache_dir,
    ordered_category_images,
    subject_metadata,
    write_json,
)


FIX_NUMBER = 14
CHANNELS = 1056
SCREEN_HEIGHT = 768
SCREEN_WIDTH = 1024


def fixation_feature(subject_id: str, images: list[str], feature_dict: dict) -> np.ndarray:
    output = -np.ones((CHANNELS, len(images), FIX_NUMBER), dtype=np.float32)
    txt_root = Path(__file__).resolve().parents[1] / "EMS" / "TXT"
    for image_index, image_name in enumerate(images):
        path = txt_root / f"{subject_id}_{image_name}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            continue
        fixation = np.loadtxt(path, delimiter=",", ndmin=2)
        image_feature = feature_dict[image_name]
        feature_height, feature_width, channels = image_feature.shape
        if channels != CHANNELS:
            raise RuntimeError(f"Unexpected feature shape for {image_name}: {image_feature.shape}")
        xs = np.floor(fixation[:, 1] / SCREEN_WIDTH * feature_width).astype(np.int32)
        ys = np.floor(fixation[:, 2] / SCREEN_HEIGHT * feature_height).astype(np.int32)
        valid = (xs >= 0) & (xs < feature_width) & (ys >= 0) & (ys < feature_height)
        xs, ys = xs[valid][:FIX_NUMBER], ys[valid][:FIX_NUMBER]
        if len(xs):
            output[:, image_index, : len(xs)] = image_feature[ys, xs, :].T
    return output


def cache_is_complete(folder: Path, subjects: pd.DataFrame, images: list[str]) -> bool:
    feature_path = folder / "features.npy"
    manifest_path = folder / "manifest.json"
    metadata_path = folder / "subjects.csv"
    if not all(path.exists() for path in [feature_path, manifest_path, metadata_path]):
        return False
    try:
        array = np.load(feature_path, mmap_mode="r")
        return array.shape == (len(subjects), CHANNELS, len(images), FIX_NUMBER)
    except Exception:
        return False


def build_one(category: str, images: list[str], feature_dict: dict, subjects: pd.DataFrame, force: bool) -> None:
    folder = cache_dir(category)
    if cache_is_complete(folder, subjects, images) and not force:
        print(f"[{category}] complete cache exists; skipping")
        return

    target = folder / "features.npy"
    partial = folder / "features.partial.npy"
    if partial.exists():
        partial.unlink()
    shape = (len(subjects), CHANNELS, len(images), FIX_NUMBER)
    print(f"[{category}] creating {shape}, {np.prod(shape) * 4 / 1024**3:.2f} GB")
    cache = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
    started = time.perf_counter()
    for index, row in tqdm(subjects.iterrows(), total=len(subjects), desc=category):
        cache[index] = fixation_feature(row["subject_id"], images, feature_dict)
        if (index + 1) % 10 == 0:
            cache.flush()
    cache.flush()
    del cache
    partial.replace(target)
    subjects.to_csv(folder / "subjects.csv", index=False)
    write_json(
        folder / "manifest.json",
        {
            "category": category,
            "image_count": len(images),
            "images": images,
            "shape": list(shape),
            "dtype": "float32",
            "fix_number": FIX_NUMBER,
            "channels": CHANNELS,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(f"[{category}] saved {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["all", *CATEGORY_ORDER], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not FEATURE_DICT_PATH.exists():
        raise FileNotFoundError(FEATURE_DICT_PATH)
    txt_count = len(list((FEATURE_DICT_PATH.parent / "EMS" / "TXT").glob("*.txt")))
    if txt_count != 20_800:
        raise RuntimeError(f"Expected 20800 TXT inputs, found {txt_count}")

    print("Loading 10.88 GB RINet feature dictionary...")
    started = time.perf_counter()
    feature_dict = np.load(FEATURE_DICT_PATH, allow_pickle=True).item()
    print(f"Loaded {len(feature_dict)} images in {time.perf_counter() - started:.1f}s")
    categories = ordered_category_images(list(feature_dict.keys()))
    subjects = subject_metadata()
    selected = CATEGORY_ORDER if args.category == "all" else (args.category,)
    for category in selected:
        build_one(category, categories[category], feature_dict, subjects, args.force)


if __name__ == "__main__":
    main()
