import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


class Protocol211Tests(unittest.TestCase):
    def test_completed_run_can_be_reopened_after_json_roundtrip(self):
        from reviewer_experiments.train_211 import main
        from reviewer_experiments.seed_alpha import sha256

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            (cache / "features_100.npy").write_bytes(b"not read for a completed run")
            behavior, features = root / "behavior.csv", root / "features.json"
            behavior.write_text("subject_int\n")
            features.write_text("{}")
            folder = root / "results" / "Set_0"
            folder.mkdir(parents=True)
            config = {"eval_fold": "Set_0", "alpha_fold": "Set_1", "train_folds": ["Set_2", "Set_3"],
                      "epochs": 50, "batch_size": 4, "seed": 42,
                      "cache_sha256": sha256(cache / "features_100.npy"),
                      "behavior_sha256": sha256(behavior), "feature_sets_sha256": sha256(features)}
            (folder / "run_config.json").write_text(json.dumps(config))
            (folder / "completed.json").write_text("{}")
            argv = ["train_211", "--cache-dir", str(cache), "--behavior-features", str(behavior),
                    "--feature-sets", str(features), "--output-root", str(root / "results"),
                    "--eval-fold", "Set_0"]
            with patch("sys.argv", argv):
                main()

    def test_four_roles_cover_each_fold_once_without_overlap(self):
        from reviewer_experiments.protocol_211 import fold_roles

        roles = [fold_roles(f"Set_{i}") for i in range(4)]
        for train, alpha, evaluation in roles:
            self.assertEqual(len(train), 2)
            self.assertEqual(len(set(train + (alpha, evaluation))), 4)
            self.assertEqual(set(train + (alpha, evaluation)), {f"Set_{i}" for i in range(4)})
        self.assertEqual([x[1] for x in roles], ["Set_1", "Set_2", "Set_3", "Set_0"])

    def test_alpha_selection_and_threshold_use_calibration_only(self):
        from reviewer_experiments.protocol_211 import select_alpha

        labels = np.array([0, 0, 1, 1])
        msnet = np.array([0.1, 0.2, 0.8, 0.9])
        behavioral = np.array([0.9, 0.8, 0.2, 0.1])
        result = select_alpha(labels, msnet, behavioral, np.array([0.0, 1.0]))
        self.assertEqual(result["alpha_msnet"], 1.0)
        self.assertGreater(result["auc"], 0.99)
        self.assertGreaterEqual(result["threshold"], 0.0)
        self.assertLessEqual(result["threshold"], 1.0)

    def test_workbook_mapping_rejects_duplicate_subjects(self):
        from reviewer_experiments.protocol_211 import validate_subject_map

        frame = pd.DataFrame({"subject_int": [1, 1, 201, 202],
                              "set_name": ["Set_0", "Set_1", "Set_2", "Set_3"],
                              "label": [0, 0, 1, 1]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_subject_map(frame, expected_per_fold=1)

    def test_eval_metrics_apply_preselected_threshold(self):
        from reviewer_experiments.train_211 import fixed_threshold_metrics

        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        result = fixed_threshold_metrics(labels, scores, threshold=0.7)
        self.assertEqual(result["acc"], 0.75)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fn"], 1)


if __name__ == "__main__":
    unittest.main()
