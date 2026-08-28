from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "EMS"
REPO = ROOT / "EMS_repo"
FEATURE_DICT = ROOT / "feature_dict_RINet.npy"
WEIGHTS = ROOT / "model_parameters"
NOTEBOOK = ROOT / "notebook_windows.ipynb"


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


for path, description in [
    (DATASET / "Train_Valid.xlsx", "fold definition workbook"),
    (DATASET / "Train_Valid" / "Fixations", "training fixation directory"),
    (DATASET / "Test" / "Fixations", "test fixation directory"),
    (DATASET / "Images", "stimulus image directory"),
    (FEATURE_DICT, "RINet feature dictionary"),
    (REPO / "MSNet.py", "official EMS source"),
    (NOTEBOOK, "Windows notebook"),
]:
    require(path, description)

counts = {
    "train_fixations": len(list((DATASET / "Train_Valid" / "Fixations").glob("*.xlsx"))),
    "test_fixations": len(list((DATASET / "Test" / "Fixations").glob("*.xlsx"))),
    "images": len([p for p in (DATASET / "Images").rglob("*") if p.is_file()]),
    "checkpoints": len(list(WEIGHTS.glob("Set_*/best_model.pt"))),
}
expected = {
    "train_fixations": 160,
    "test_fixations": 48,
    "images": 100,
    "checkpoints": 4,
}
if counts != expected:
    raise RuntimeError(f"Dataset counts differ from expected values: {counts} != {expected}")

required_modules = [
    "torch",
    "numpy",
    "pandas",
    "sklearn",
    "openpyxl",
    "cv2",
    "matplotlib",
    "tqdm",
    "umap",
    "lime",
    "IPython",
]
for name in required_modules:
    importlib.import_module(name)

import torch

sys.path.insert(0, str(REPO))
from MSNet import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = WEIGHTS / "Set_0" / "best_model.pt"
state = torch.load(checkpoint_path, map_location=device, weights_only=True)
state = {key.removeprefix("module."): value for key, value in state.items()}
model = Model(in_c=1056).to(device)
model.load_state_dict(state)
model.eval()
with torch.inference_mode():
    sample = torch.zeros((1, 1056, 100, 14), dtype=torch.float32, device=device)
    prediction = model(sample).item()

with NOTEBOOK.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)
if not notebook.get("cells"):
    raise RuntimeError("The generated Windows notebook contains no cells")

txt_dir = DATASET / "TXT"
summary = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "model_smoke_prediction": prediction,
    "feature_dict_size_gb": round(FEATURE_DICT.stat().st_size / 1024**3, 2),
    "txt_files": len(list(txt_dir.glob("*.txt"))) if txt_dir.exists() else 0,
    **counts,
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
