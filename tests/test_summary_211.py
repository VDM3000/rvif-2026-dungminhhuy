import unittest

import pandas as pd


class Summary211Tests(unittest.TestCase):
    def test_audit_rejects_stale_metric(self):
        from reviewer_experiments.summarize_211 import audit_prediction_metrics
        from reviewer_experiments.train_211 import fixed_threshold_metrics

        part = pd.DataFrame({"label": [0, 0, 1, 1], "fused_prob": [0.1, 0.2, 0.8, 0.9],
                             "threshold": [0.5] * 4, "pred": [0, 0, 1, 1]})
        detail = fixed_threshold_metrics(part.label, part.fused_prob, 0.5)
        self.assertEqual(audit_prediction_metrics(part, detail, "fused_prob")["auc"], 1.0)
        detail["auc"] = 0.5
        with self.assertRaisesRegex(ValueError, "Saved metric mismatch: auc"):
            audit_prediction_metrics(part, detail, "fused_prob")

    def test_audit_rejects_stale_prediction(self):
        from reviewer_experiments.summarize_211 import audit_prediction_metrics
        from reviewer_experiments.train_211 import fixed_threshold_metrics

        part = pd.DataFrame({"label": [0, 0, 1, 1], "fused_prob": [0.1, 0.2, 0.8, 0.9],
                             "threshold": [0.5] * 4, "pred": [1, 0, 1, 1]})
        detail = fixed_threshold_metrics(part.label, part.fused_prob, 0.5)
        with self.assertRaisesRegex(ValueError, "Predicted label mismatch"):
            audit_prediction_metrics(part, detail, "fused_prob")

    def test_audit_rejects_duplicate_evaluated_subject(self):
        from reviewer_experiments.summarize_211 import audit_evaluations

        expected = pd.DataFrame({"subject_int": [1, 2, 201, 202],
                                 "set_name": ["Set_0", "Set_1", "Set_2", "Set_3"],
                                 "label": [0, 0, 1, 1]})
        actual = expected.copy()
        actual.loc[3, "subject_int"] = 201
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_evaluations(actual, expected)

    def test_audit_rejects_wrong_label_or_fold(self):
        from reviewer_experiments.summarize_211 import audit_evaluations

        expected = pd.DataFrame({"subject_int": [1, 201], "set_name": ["Set_0", "Set_1"], "label": [0, 1]})
        actual = expected.copy()
        actual.loc[1, "label"] = 0
        with self.assertRaisesRegex(ValueError, "mismatch"):
            audit_evaluations(actual, expected)

    def test_msnet_threshold_is_selected_from_calibration_scores(self):
        from reviewer_experiments.summarize_211 import msnet_from_calibration

        calibration = pd.DataFrame({"subject_int": [1, 2, 201, 202],
                                    "label": [0, 0, 1, 1],
                                    "msnet_prob": [0.1, 0.2, 0.8, 0.9]})
        evaluated = pd.DataFrame({"subject_int": [3, 4, 203, 204],
                                  "set_name": ["Set_0"] * 4, "label": [0, 0, 1, 1],
                                  "msnet_prob": [0.3, 0.4, 0.6, 0.7]})
        predictions, metric = msnet_from_calibration(calibration, evaluated)
        self.assertEqual(predictions.threshold.nunique(), 1)
        self.assertAlmostEqual(predictions.threshold.iloc[0], 0.2, delta=0.01)
        self.assertEqual(metric["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
