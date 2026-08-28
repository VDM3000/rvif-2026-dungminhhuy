from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "EMS_repo"
sys.path.insert(0, str(REPO))

from dataloader import TestLoader
from MSNet import Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = SimpleNamespace(
    dataset_dir=str(ROOT / "EMS"),
    fix_number=14,
    fix_embedding=1056,
)

started = time.perf_counter()
feature_dict = np.load(ROOT / "feature_dict_RINet.npy", allow_pickle=True).item()
feature_load_seconds = time.perf_counter() - started

folds = pd.read_excel(ROOT / "EMS" / "Train_Valid.xlsx")
subject_id = str(int(folds["Set_0"].dropna().iloc[0])).zfill(3)
dataset = TestLoader(set_ids=[subject_id], feature_dict=feature_dict, args=args)
loaded_id, feature = dataset[0]

checkpoint = torch.load(
    ROOT / "model_parameters" / "Set_0" / "best_model.pt",
    map_location=device,
    weights_only=True,
)
checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
model = Model(in_c=1056).to(device)
model.load_state_dict(checkpoint)
model.eval()

started = time.perf_counter()
with torch.inference_mode():
    tensor = torch.from_numpy(feature).unsqueeze(0).to(device)
    probability = model(tensor).item()
inference_seconds = time.perf_counter() - started

print(f"subject_id={loaded_id}")
print(f"feature_shape={feature.shape}")
print(f"feature_load_seconds={feature_load_seconds:.2f}")
print(f"inference_seconds={inference_seconds:.3f}")
print(f"probability={probability:.9f}")
print(f"device={device}")
