# MSNet Baseline Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an isolated workspace that can reproduce the EMS authors' four-fold MSNet validation with their released checkpoints, without modifying upstream code or existing BAMF files.

**Architecture:** A clean clone of YingjieSong1/EMS lives under `msnet_reproduction/EMS_repo`. Local input links point to the existing EMS dataset and RINet feature dictionary; the four released checkpoints are copied into a separate result root where upstream validation scripts write their outputs.

**Tech Stack:** Git, PowerShell, Python, PyTorch, pandas, EMS upstream scripts.

**Spec:** User request on 2026-11-23 and [EMS README](https://github.com/YingjieSong1/EMS#dataset-ems).

## Global Constraints

- Preserve the existing `EMS_repo/`, notebooks, data, and manuscript.
- Keep the new EMS clone at upstream commit `966eef1669623db16f69aabc1db8b22299459b00` for comparison with the prior baseline.
- Do not copy overrides into the new EMS clone.
- Do not duplicate the 11-GiB RINet dictionary.
- Do not call validation a test-set result; test labels are unavailable locally.

## Review Focus

- Confirm the clone is clean and from the intended upstream URL and commit.
- Confirm four checkpoint copies are byte-identical to their sources.
- Confirm local data links resolve to the expected EMS files and RINet dictionary.
- Confirm workspace artifacts are ignored by the parent Git repository unless intentionally documented.
- Confirm whether upstream `val.py` runs in the current Python environment before claiming reproduction.

---

### Task 1: Create clean upstream source

- [ ] Clone upstream EMS into `msnet_reproduction/EMS_repo`.
- [ ] Check out the pinned commit and verify `git status --short` is empty.

### Task 2: Prepare inputs and outputs

- [ ] Link `msnet_reproduction/data/EMS` to the existing dataset.
- [ ] Link `msnet_reproduction/data/feature_dict_RINet.npy` to the existing dictionary.
- [ ] Copy each `model_parameters/Set_i/best_model.pt` into `msnet_reproduction/results/msnet/Set_i/` and compare hashes.

### Task 3: Document and verify the run boundary

- [ ] Write `msnet_reproduction/README.md` with exact upstream commands and local paths.
- [ ] Check directory contents and Git ignore rules.
- [ ] Run a lightweight environment check; report any compatibility blocker before changing upstream code.
