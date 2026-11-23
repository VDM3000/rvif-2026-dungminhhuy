"""Convert the large EMS image-feature dictionary to a 160-subject mmap cache."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .splits import load_subject_map
from .seed_analysis import sha256


SHAPE = (160, 1056, 100, 14)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    dataset = root / "EMS"
    feature_path = root / "feature_dict_RINet.npy"
    expected_bytes = np.prod(SHAPE) * np.dtype("float32").itemsize
    if (args.output / "features_100.npy").exists():
        raise FileExistsError(f"Cache already exists: {args.output / 'features_100.npy'}")
    if shutil.disk_usage(args.output.parent if args.output.parent.exists() else root).free < expected_bytes + 512 * 1024**2:
        raise RuntimeError(f"Need at least {(expected_bytes + 512 * 1024**2)/1024**3:.2f} GiB free for cache")
    metadata = load_subject_map(dataset / "Train_Valid.xlsx")
    if len(metadata) != SHAPE[0]:
        raise ValueError("Expected 160 subjects")
    sys.path.insert(0, str(root / "EMS_repo"))
    from dataloader import TestLoader

    print("Loading RINet dictionary once", flush=True)
    feature_dict = np.load(feature_path, allow_pickle=True).item()
    if len(feature_dict) != 100:
        raise ValueError(f"Expected 100 stimulus features, found {len(feature_dict)}")
    args.output.mkdir(parents=True, exist_ok=True)
    temp = args.output / "features_100.partial.npy"
    if temp.exists():
        raise FileExistsError(f"Incomplete cache exists; inspect before retrying: {temp}")
    mem = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=SHAPE)
    loader = TestLoader(set_ids=metadata.subject_id.tolist(), feature_dict=feature_dict,
                        args=SimpleNamespace(dataset_dir=str(dataset), fix_number=14, fix_embedding=1056))
    for index in range(len(loader)):
        subject, array = loader[index]
        if subject != metadata.iloc[index].subject_id or array.shape != SHAPE[1:]:
            raise RuntimeError(f"Cache subject/shape mismatch at row {index}: {subject}, {array.shape}")
        mem[index] = array
        if (index + 1) % 20 == 0:
            mem.flush()
            print(f"cached {index + 1}/160", flush=True)
    mem.flush()
    del mem
    temp.replace(args.output / "features_100.npy")
    metadata.to_csv(args.output / "subjects.csv", index=False)
    (args.output / "manifest.json").write_text(json.dumps({
        "shape": SHAPE, "dtype": "float32", "subject_count": 160,
        "feature_dict_sha256": sha256(feature_path),
        "workbook_sha256": sha256(dataset / "Train_Valid.xlsx"),
        "source": "EMS_repo/dataloader.py TestLoader, 100 images, 14 fixations, 1056 channels",
    }, indent=2), encoding="utf-8")
    print(f"Ready: {args.output / 'features_100.npy'}", flush=True)


if __name__ == "__main__":
    main()
