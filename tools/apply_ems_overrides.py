"""Apply the four documented EMS compatibility changes on any OS.

The pinned submodule remains the upstream reference. The four working-tree
changes are allowed only when the target matches upstream or our override.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = ROOT / "EMS_repo"
OVERRIDES = ROOT / "ems_overrides"
NAMES = ("MSNet.py", "train.py", "val.py", "test.py")


def main() -> None:
    if not (SUBMODULE / ".git").exists():
        raise RuntimeError("EMS_repo is missing; run git submodule update --init --recursive")
    pinned = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SUBMODULE,
                            capture_output=True, text=True, check=True).stdout.strip()
    expected = subprocess.run(["git", "rev-parse", ":EMS_repo"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    if pinned != expected:
        raise RuntimeError(f"Submodule is at {pinned}, expected pinned commit {expected}")
    for name in NAMES:
        destination = SUBMODULE / name
        override = (OVERRIDES / name).read_bytes()
        original = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=SUBMODULE,
                                  capture_output=True, check=True).stdout
        current = destination.read_bytes()
        if current not in (original, override):
            raise RuntimeError(f"Refusing to replace an unrelated local edit: {destination}")
    for name in NAMES:
        destination = SUBMODULE / name
        data = (OVERRIDES / name).read_bytes()
        destination.write_bytes(data)
        print(f"{name}: {hashlib.sha256(data).hexdigest()}")
    print(f"Applied four compatibility overrides to pinned EMS {pinned}. Local submodule is now intentionally modified.")


if __name__ == "__main__":
    main()
