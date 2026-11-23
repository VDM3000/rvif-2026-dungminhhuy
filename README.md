# BAMF

BAMF combines MSNet with pupil and scanpath features for schizophrenia recognition on the EMS dataset.

## Setup

```bash
git clone --recurse-submodules https://github.com/VDM3000/rvif-2026-dungminhhuy.git
cd rvif-2026-dungminhhuy
python -m venv .venv
```

Activate the environment and install PyTorch 2.7.0 and torchvision 0.22.0 with CUDA 12.8, then install the project dependencies.

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python tools/apply_ems_overrides.py
```

Download the following files from the [official EMS repository](https://github.com/YingjieSong1/EMS#dataset-ems). See its [Train section](https://github.com/YingjieSong1/EMS#train) for the feature maps and model weights.

| Download | Location |
|---|---|
| EMS dataset | EMS/ (including Train_Valid.xlsx, Train_Valid/Fixations/, Test/Fixations/, and Images/) |
| RINet feature dictionary | feature_dict_RINet.npy |
| Pretrained MSNet weights | model_parameters/Set_0/best_model.pt through model_parameters/Set_3/best_model.pt |

## Reproduce

Run [the validation notebook](notebooks/windows.ipynb) with the project environment to generate the MSNet scores and behavioral features under outputs/improve_behavioral/. [A Kaggle version](notebooks/original_kaggle.ipynb) is also available.

```bash
python scripts/reproduce_paper.py all
```

This runs the seed analysis, linear Early Fusion on the four development folds, and the 2/1/1 experiments. Outputs are written to work/paper_reproduction/. Run with --help to select a stage.

The first run uses `bamf/preprocess.py` to extract fixation features from the RINet dictionary and store a reusable local cache.

For the stimulus experiments, run the following after the validation notebook:

```bash
python category_experiments/preprocess.py --category all
python category_experiments/train_msnet.py --category all
python category_experiments/evaluate.py
python balanced_stimulus_experiment/prepare.py
python balanced_stimulus_experiment/train_msnet.py
python balanced_stimulus_experiment/evaluate.py
```
