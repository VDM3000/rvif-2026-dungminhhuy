from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CATEGORY_ORDER, FULL_RESULTS_DIR
from run_fixed_protocol import ALPHA_MSNET, BEHAVIOR_WEIGHT, OUTPUT_ROOT


def main() -> None:
    checks: dict[str, object] = {}
    mapping = pd.read_csv(OUTPUT_ROOT / "stimulus_category_mapping.csv")
    checks["mapping_rows"] = len(mapping)
    checks["mapping_unique_images"] = int(mapping["image_stem"].nunique())
    assert len(mapping) == 100 and mapping["image_stem"].nunique() == 100

    stimulus = pd.read_csv(FULL_RESULTS_DIR / "stim_behavior_features.csv", usecols=["image"])
    behavior_images = {Path(str(value)).stem for value in stimulus["image"].unique()}
    assert behavior_images == set(mapping["image_stem"])
    checks["mapping_matches_behavior_table"] = True

    category_checks = {}
    for category in CATEGORY_ORDER:
        folder = OUTPUT_ROOT / category
        predictions = pd.read_csv(folder / "subject_predictions.csv", dtype={"subject_id": str})
        features = json.loads((folder / "feature_columns.json").read_text(encoding="utf-8"))
        fold_counts = predictions.groupby("fold").size().to_dict()
        assert len(predictions) == 160
        assert predictions["subject_int"].nunique() == 160
        assert set(fold_counts) == {"Set_0", "Set_1", "Set_2", "Set_3"}
        assert all(count == 40 for count in fold_counts.values())
        assert all("duration_" not in feature for feature in features)
        assert any(feature.startswith("pupil_") for feature in features)
        expected = ALPHA_MSNET * predictions["p_msnet"] + BEHAVIOR_WEIGHT * predictions["p_behavior"]
        assert np.allclose(predictions["p_fusion"], expected, rtol=0, atol=1e-12)
        assert predictions[["p_msnet", "p_behavior", "p_fusion"]].apply(
            lambda column: column.between(0, 1).all()
        ).all()
        expected_label = (predictions["p_fusion"] >= predictions["threshold"]).astype(int)
        assert np.array_equal(expected_label, predictions["predicted_label"].astype(int))
        for name in ("msnet", "behavior", "fusion"):
            metrics = pd.read_csv(folder / f"{name}_foldwise_metrics.csv")
            assert list(metrics["set_name"]) == ["Set_0", "Set_1", "Set_2", "Set_3", "average"]
        category_checks[category] = {
            "subjects": len(predictions),
            "unique_subjects": int(predictions["subject_int"].nunique()),
            "fold_counts": fold_counts,
            "pupil_scanpath_features": len(features),
            "fusion_formula_exact": True,
            "predicted_labels_match_fold_thresholds": True,
        }
    checks["categories"] = category_checks
    checks["status"] = "PASS"
    output = OUTPUT_ROOT / "validation_report.json"
    output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
