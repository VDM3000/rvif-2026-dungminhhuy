from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    CACHE_PATH,
    CATEGORY_CACHE_ROOT,
    CATEGORY_NAMES,
    CATEGORY_ORDER,
    MANIFEST_ROOT,
    MASTER_MANIFEST_PATH,
    SEED,
    STIMULUS_COUNTS,
    SUBJECTS_PATH,
    WORK_ROOT,
    manifest_path,
    write_json,
)


def load_category_sources() -> tuple[dict[str, dict], dict[str, np.ndarray], pd.DataFrame]:
    manifests: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    subjects: pd.DataFrame | None = None
    for category in CATEGORY_ORDER:
        folder = CATEGORY_CACHE_ROOT / category
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        array = np.load(folder / "features.npy", mmap_mode="r")
        metadata = pd.read_csv(folder / "subjects.csv", dtype={"subject_id": str})
        if array.shape != (160, 1056, len(manifest["images"]), 14):
            raise RuntimeError(f"Invalid category cache for {category}: {array.shape}")
        if subjects is None:
            subjects = metadata
        elif not metadata[["subject_int", "set_name", "label"]].equals(
            subjects[["subject_int", "set_name", "label"]]
        ):
            raise RuntimeError(f"Subject metadata differs for {category}")
        manifests[category] = manifest
        arrays[category] = array
    assert subjects is not None
    return manifests, arrays, subjects


def shuffled_images(manifests: dict[str, dict]) -> dict[str, list[str]]:
    rng = np.random.default_rng(SEED)
    result: dict[str, list[str]] = {}
    for category in CATEGORY_ORDER:
        images = np.asarray(sorted(manifests[category]["images"]), dtype=object)
        result[category] = rng.permutation(images).tolist()
    return result


def round_robin_master(shuffled: dict[str, list[str]]) -> list[tuple[str, str]]:
    # Fifteen complete rounds make every prefix 20/40/60 exactly balanced.
    return [
        (category, shuffled[category][rank])
        for rank in range(15)
        for category in CATEGORY_ORDER
    ]


def image_filename(category: str, stem: str) -> str:
    folder = Path(__file__).resolve().parents[1] / "EMS" / "Images" / CATEGORY_NAMES[category]
    matches = [path.name for path in folder.iterdir() if path.is_file() and path.stem == stem]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one image file for {category}/{stem}, found {matches}")
    return matches[0]


def write_manifests(master: list[tuple[str, str]]) -> None:
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    for count in STIMULUS_COUNTS:
        rows = [
            {"IMAGE": image_filename(category, stem), "CATEGORY": CATEGORY_NAMES[category]}
            for category, stem in master[:count]
        ]
        frame = pd.DataFrame(rows)
        frame.to_csv(manifest_path(count), index=False)
        observed = frame["CATEGORY"].value_counts().to_dict()
        if set(observed.values()) != {count // 4}:
            raise RuntimeError(f"Unbalanced manifest {count}: {observed}")


def write_master_cache(
    master: list[tuple[str, str]],
    manifests: dict[str, dict],
    arrays: dict[str, np.ndarray],
    subjects: pd.DataFrame,
    force: bool,
) -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists() and not force:
        existing = np.load(CACHE_PATH, mmap_mode="r")
        if existing.shape == (160, 1056, 60, 14):
            print(f"Complete master cache exists; skipping {CACHE_PATH}")
            subjects.to_csv(SUBJECTS_PATH, index=False)
            return
    partial = CACHE_PATH.with_suffix(".partial.npy")
    if partial.exists():
        partial.unlink()
    cache = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(160, 1056, 60, 14)
    )
    for target_index, (category, stem) in enumerate(master):
        source_index = manifests[category]["images"].index(stem)
        cache[:, :, target_index, :] = arrays[category][:, :, source_index, :]
        print(f"[{target_index + 1:02d}/60] {category}/{stem}")
    cache.flush()
    del cache
    partial.replace(CACHE_PATH)
    subjects.to_csv(SUBJECTS_PATH, index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifests, arrays, subjects = load_category_sources()
    shuffled = shuffled_images(manifests)
    master = round_robin_master(shuffled)
    write_manifests(master)
    write_master_cache(master, manifests, arrays, subjects, args.force)
    write_json(
        MASTER_MANIFEST_PATH,
        {
            "seed": SEED,
            "selection": "lexicographic category lists, one sequential NumPy RNG, round-robin prefixes",
            "category_order": list(CATEGORY_ORDER),
            "stimulus_counts": list(STIMULUS_COUNTS),
            "master_order": [
                {"position": index, "category": category, "image_stem": stem}
                for index, (category, stem) in enumerate(master)
            ],
            "cache_shape": [160, 1056, 60, 14],
        },
    )
    print("Prepared balanced nested manifests and master cache.")


if __name__ == "__main__":
    main()
