from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ALPHA_MSNET,
    BEHAVIOR_WEIGHT,
    CACHE_PATH,
    CATEGORY_NAMES,
    FOLDS,
    MASTER_MANIFEST_PATH,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    STIMULUS_COUNTS,
    load_manifest,
    result_dir,
    write_json,
)


def main():
    report: dict[str, object] = {}
    manifests = {count: load_manifest(count) for count in STIMULUS_COUNTS}
    expected_categories = set(CATEGORY_NAMES.values())
    for count, frame in manifests.items():
        assert len(frame) == count
        assert frame["image_stem"].nunique() == count
        counts = frame["CATEGORY"].value_counts().to_dict()
        assert set(counts) == expected_categories
        assert set(counts.values()) == {count // 4}
    assert set(manifests[20]["image_stem"]) < set(manifests[40]["image_stem"])
    assert set(manifests[40]["image_stem"]) < set(manifests[60]["image_stem"])
    master = json.loads(MASTER_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert master["seed"] == 42
    assert np.load(CACHE_PATH, mmap_mode="r").shape == (160, 1056, 60, 14)

    master_stems = [row["image_stem"] for row in master["master_order"]]
    for count in STIMULUS_COUNTS:
        assert list(manifests[count]["image_stem"]) == master_stems[:count]

    behavior_source = pd.read_csv(
        PROJECT_ROOT / "improve_behavioral_final_results" / "stim_behavior_features.csv",
        usecols=["subject_int", "image"],
    )
    behavior_source["image_stem"] = behavior_source["image"].map(
        lambda value: Path(str(value)).stem
    )
    behavior_audit = {}
    for count in STIMULUS_COUNTS:
        selected = set(manifests[count]["image_stem"])
        filtered = behavior_source[behavior_source["image_stem"].isin(selected)]
        assert set(filtered["image_stem"].unique()) == selected
        assert filtered["subject_int"].nunique() == 160
        assert filtered.groupby("subject_int")["image_stem"].nunique().max() <= count
        behavior_audit[str(count)] = {
            "selected_unique_images": int(filtered["image_stem"].nunique()),
            "subjects": int(filtered["subject_int"].nunique()),
            "outside_manifest_images": 0,
        }

    condition_reports = {}
    all_subjects = None
    for count in STIMULUS_COUNTS:
        predictions = pd.read_csv(result_dir(count) / "subject_predictions.csv", dtype={"subject_id": str})
        assert len(predictions) == 160 and predictions["subject_int"].nunique() == 160
        assert predictions["stimulus_count"].eq(count).all()
        fold_counts = predictions["fold"].value_counts().to_dict()
        assert set(fold_counts) == set(FOLDS) and set(fold_counts.values()) == {40}
        subjects = predictions[["subject_int", "fold", "true_label"]].sort_values("subject_int").reset_index(drop=True)
        if all_subjects is None:
            all_subjects = subjects
        else:
            assert subjects.equals(all_subjects)
        expected = ALPHA_MSNET * predictions["p_msnet"] + BEHAVIOR_WEIGHT * predictions["p_behavior"]
        assert np.allclose(predictions["p_fusion"], expected, rtol=0, atol=1e-12)
        assert predictions[["p_msnet", "p_behavior", "p_fusion"]].apply(
            lambda column: column.between(0, 1).all()
        ).all()
        expected_label = (predictions["p_fusion"] >= predictions["threshold"]).astype(int)
        assert np.array_equal(expected_label, predictions["predicted_label"].astype(int))
        features = json.loads((result_dir(count) / "feature_columns.json").read_text(encoding="utf-8"))
        assert len(features) == 173
        assert any(feature.startswith("pupil_") for feature in features)
        assert not any(feature.startswith("duration_") for feature in features)
        metrics = pd.read_csv(result_dir(count) / "fold_metrics.csv")
        assert set(metrics["component"]) == {"msnet", "behavior", "fusion"}
        assert len(metrics) == 15
        completed = list((result_dir(count) / "msnet").glob("Set_*/completed.json"))
        assert len(completed) == 4
        condition_reports[str(count)] = {
            "images_per_category": count // 4,
            "subjects": 160,
            "fold_counts": fold_counts,
            "feature_count": 173,
            "fusion_formula_exact": True,
            "completed_msnet_folds": 4,
        }
    report.update({
        "status": "PASS",
        "nested": "S20 subset S40 subset S60",
        "cache_shape": [160, 1056, 60, 14],
        "behavior_input_audit": behavior_audit,
        "conditions": condition_reports,
    })
    write_json(OUTPUT_ROOT / "validation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
