from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebookaaf7c6b835.ipynb"
TARGET = ROOT / "notebook_windows.ipynb"


def source_lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def set_cell(notebook: dict, index: int, text: str) -> None:
    notebook["cells"][index]["source"] = source_lines(text)
    notebook["cells"][index]["outputs"] = []
    notebook["cells"][index]["execution_count"] = None


with SOURCE.open("r", encoding="utf-8") as handle:
    nb = json.load(handle)

set_cell(
    nb,
    2,
    r'''
import os
import sys
import glob
import torch
from pathlib import Path


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "EMS").is_dir() and (candidate / "EMS_repo").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot locate the project root containing EMS and EMS_repo. "
        "Start Jupyter from the RVIF_2026_Experiment directory."
    )


PROJECT_ROOT = find_project_root()
WORK_DIR = str(PROJECT_ROOT)
REPO_DIR = str(PROJECT_ROOT / "EMS_repo")
DATASET_DIR = str(PROJECT_ROOT / "EMS")
FEATURE_DICT = str(PROJECT_ROOT / "feature_dict_RINet.npy")
MODEL_SRC = str(PROJECT_ROOT / "model_parameters")
OUT_DIR = str(PROJECT_ROOT / "outputs" / "improve_behavioral")
BASELINE_BACKUP_DIR = str(PROJECT_ROOT / "work" / "result_pretrained_backup")

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "work").mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Python:", sys.executable)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Dataset:", DATASET_DIR)
print("Feature dictionary:", FEATURE_DICT)
print("Feature size (GB):", round(Path(FEATURE_DICT).stat().st_size / 1024**3, 2))
print("Windows safety: DataLoader workers are fixed at 0 to avoid duplicating the 11 GB feature dictionary.")
''',
)

set_cell(
    nb,
    4,
    r'''
import os
import sys
from pathlib import Path

required_repo_files = [
    "MSNet.py",
    "val.py",
    "test.py",
    "data_process/split_fix.py",
    "data_process/metrics_val.py",
    "data_process/metrics_ostu.py",
]

print("Using the local official EMS repository:", REPO_DIR)
for relative in required_repo_files:
    path = Path(REPO_DIR) / relative
    print(relative, "->", path.exists())
    assert path.exists(), f"Missing EMS source file: {path}"

venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    print("WARNING: select the 'RVIF EMS (Windows GPU)' kernel before continuing.")
''',
)

set_cell(
    nb,
    5,
    r'''
import os
import shutil
from pathlib import Path

RESULT_DIR = str(PROJECT_ROOT / "work" / "msnet_results")

required_paths = {
    "RINet feature dictionary": Path(FEATURE_DICT),
    "pretrained model directory": Path(MODEL_SRC),
    "EMS dataset": Path(DATASET_DIR),
    "official EMS repository": Path(REPO_DIR),
    "fold workbook": Path(DATASET_DIR) / "Train_Valid.xlsx",
}
for name, path in required_paths.items():
    print(name, "->", path, "->", path.exists())
    assert path.exists(), f"Missing {name}: {path}"

# Copy only the small checkpoints to a writable result directory. The 11 GB
# feature dictionary and the dataset remain in place and are never duplicated.
for set_name in ["Set_0", "Set_1", "Set_2", "Set_3"]:
    source_weight = Path(MODEL_SRC) / set_name / "best_model.pt"
    target_dir = Path(RESULT_DIR) / set_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_weight = target_dir / "best_model.pt"
    assert source_weight.exists(), f"Missing pretrained checkpoint: {source_weight}"
    if not target_weight.exists() or target_weight.stat().st_size != source_weight.stat().st_size:
        shutil.copy2(source_weight, target_weight)

print("Local setup completed without copying the dataset or feature dictionary.")
print("Writable MSNet results:", RESULT_DIR)
''',
)

set_cell(
    nb,
    7,
    r'''
import os
import sys
import glob
import subprocess
from pathlib import Path

TXT_DIR = str(Path(DATASET_DIR) / "TXT")
expected_txt_count = 208 * 100
existing_txt = list(Path(TXT_DIR).glob("*.txt")) if Path(TXT_DIR).exists() else []

if len(existing_txt) != expected_txt_count:
    print(f"Generating fixation TXT files ({len(existing_txt)}/{expected_txt_count} currently present)...")
    cmd = [
        sys.executable,
        str(Path(REPO_DIR) / "data_process" / "split_fix.py"),
        "--dataset_dir", DATASET_DIR,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
else:
    print("TXT inputs already complete; skipping regeneration.")

txt_files = sorted(glob.glob(str(Path(TXT_DIR) / "*.txt")))
assert len(txt_files) == expected_txt_count, (
    f"Expected {expected_txt_count} TXT files, found {len(txt_files)}"
)
print("TXT directory:", TXT_DIR)
print("Number of TXT files:", len(txt_files))
print("First 10 TXT files:")
for path in txt_files[:10]:
    print(path)
''',
)

set_cell(
    nb,
    8,
    r'''
import os
import sys
import shutil
import glob
import subprocess
from pathlib import Path

RESULT_DIR = str(PROJECT_ROOT / "work" / "msnet_results")
BASELINE_BACKUP_DIR = str(PROJECT_ROOT / "work" / "result_pretrained_backup")

for set_name in ["Set_0", "Set_1", "Set_2", "Set_3"]:
    print(f"========== VALIDATE PRETRAINED {set_name} ==========")
    cmd = [
        sys.executable,
        str(Path(REPO_DIR) / "val.py"),
        "--dataset_dir", DATASET_DIR,
        "--feature_dict", FEATURE_DICT,
        "--result_root", RESULT_DIR,
        "--set_name", set_name,
        "--no_workers", "0",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

subprocess.run(
    [sys.executable, str(Path(REPO_DIR) / "data_process" / "metrics_ostu.py"), "--result_root", RESULT_DIR],
    cwd=PROJECT_ROOT,
    check=True,
)
subprocess.run(
    [sys.executable, str(Path(REPO_DIR) / "data_process" / "metrics_val.py"), "--result_root", RESULT_DIR],
    cwd=PROJECT_ROOT,
    check=True,
)

summary_path = Path(RESULT_DIR) / "val.txt"
print("\nValidation summary:")
print(summary_path.read_text(encoding="utf-8"))

if Path(BASELINE_BACKUP_DIR).exists():
    shutil.rmtree(BASELINE_BACKUP_DIR)
shutil.copytree(RESULT_DIR, BASELINE_BACKUP_DIR)

print("Backed up baseline results to:", BASELINE_BACKUP_DIR)
for path in sorted(glob.glob(str(Path(BASELINE_BACKUP_DIR) / "Set_*" / "prob_val.txt"))):
    print(path)
''',
)

replacements = {
    '`/kaggle/working/improve_behavioral_final_results.zip`': '`outputs/improve_behavioral_final_results.zip`',
    'Clone the official EMS repository and prepare writable paths under `/kaggle/working`.': 'Use the local official EMS repository and prepare writable Windows paths.',
    'OUT_DIR = "/kaggle/working/improve_behavioral"': 'OUT_DIR = str(PROJECT_ROOT / "outputs" / "improve_behavioral")',
    'REPO_DIR = "/kaggle/working/EMS_repo"': 'REPO_DIR = str(PROJECT_ROOT / "EMS_repo")',
    'DATASET_DIR = "/kaggle/working/EMS_dataset/EMS"': 'DATASET_DIR = str(PROJECT_ROOT / "EMS")',
    'BASELINE_BACKUP_DIR = "/kaggle/working/EMS_repo/result_pretrained_backup"': 'BASELINE_BACKUP_DIR = str(PROJECT_ROOT / "work" / "result_pretrained_backup")',
    'BASELINE_RESULT_DIR = f"{REPO_DIR}/result_pretrained_backup"': 'BASELINE_RESULT_DIR = str(PROJECT_ROOT / "work" / "result_pretrained_backup")',
    'PACKAGE_BASE = "/kaggle/working/improve_behavioral_final_results"': 'PACKAGE_BASE = str(PROJECT_ROOT / "outputs" / "improve_behavioral_final_results")',
    'f"{REPO_DIR}/feature_dict_RINet.npy"': 'FEATURE_DICT',
    '"--no_workers", "2"': '"--no_workers", "0"',
    '    "test.py",': '    str(Path(REPO_DIR) / "test.py"),',
    'subprocess.run(cmd, cwd=REPO_DIR, check=True)': 'subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)',
}

for index, cell in enumerate(nb["cells"]):
    if index in {2, 4, 5, 7, 8}:
        continue
    text = "".join(cell.get("source", []))
    for old, new in replacements.items():
        text = text.replace(old, new)
    cell["source"] = source_lines(text)
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

nb.setdefault("metadata", {})["kernelspec"] = {
    "display_name": "RVIF EMS (Windows GPU)",
    "language": "python",
    "name": "rvif-ems",
}
nb["metadata"].pop("kaggle", None)

with TARGET.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(nb, handle, ensure_ascii=False, indent=1)
    handle.write("\n")

print(f"Generated {TARGET}")
