import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "improve_behavioral"


class SeedAlphaParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from reviewer_experiments.seed_alpha import alpha_curve, run_seed

        cls.alpha_curve = staticmethod(alpha_curve)
        cls.run_seed = staticmethod(run_seed)
        cls.features = pd.read_csv(SOURCE / "subject_behavior_features.csv")
        cls.msnet = pd.read_csv(SOURCE / "msnet_prob_val.csv")
        cls.columns = json.loads((SOURCE / "feature_sets_clean.json").read_text(encoding="utf-8"))["pupil_scanpath"]

    def test_seed_42_matches_saved_subject_predictions(self):
        self.assertEqual(len(self.columns), 173)
        predictions, fold_metrics, alpha_by_fold = self.run_seed(
            self.features, self.msnet, self.columns, seed=42
        )
        saved = pd.read_csv(SOURCE / "best_fusion_prob_val.csv")
        predictions = predictions.sort_values("subject_int").reset_index(drop=True)
        saved = saved.sort_values("subject_int").reset_index(drop=True)
        self.assertEqual(len(predictions), 160)
        self.assertEqual(predictions.subject_int.nunique(), 160)
        self.assertEqual(predictions.groupby("set_name").size().to_dict(), {f"Set_{i}": 40 for i in range(4)})
        np.testing.assert_allclose(predictions.behavior_prob, saved.behavior_prob, atol=1e-12, rtol=0)
        np.testing.assert_allclose(predictions.fused_prob, saved.fused_prob, atol=1e-12, rtol=0)
        self.assertEqual(alpha_by_fold.chosen_alpha_msnet.tolist(), [0.8] * 4)
        self.assertAlmostEqual(fold_metrics.loc[fold_metrics.set_name == "average", "auc"].iloc[0], 0.9502533143939393, places=12)

    def test_alpha_curve_matches_saved_curve_and_msnet_baseline(self):
        predictions, _, _ = self.run_seed(self.features, self.msnet, self.columns, seed=42)
        curve = self.alpha_curve(predictions, np.round(np.arange(0.0, 1.0001, 0.05), 2))
        saved = pd.read_csv(SOURCE / "report_outputs" / "alpha_grid_auc_curve.csv")
        check = curve[curve.alpha_msnet.isin([0.0, 0.8, 1.0])].sort_values(["alpha_msnet", "set_name"])
        expected = saved[saved.alpha_msnet.isin([0.0, 0.8, 1.0]) & saved.set_name.isin([f"Set_{i}" for i in range(4)])].sort_values(["alpha_msnet", "set_name"])
        np.testing.assert_allclose(check.auc, expected.auc, atol=1e-12, rtol=0)
        self.assertEqual(len(curve), 21 * 4)
        self.assertTrue((curve.loc[curve.alpha_msnet == 1.0, "delta_auc_vs_msnet"] == 0.0).all())

    def test_identical_seed_rows_have_exact_zero_variance(self):
        from reviewer_experiments.seed_alpha import summarize_seed_metrics

        rows = pd.DataFrame({"seed": [42, 17, 2026], "auc": [0.9502533143939393] * 3})
        summary = summarize_seed_metrics(rows)
        self.assertEqual(summary.loc["auc", "sample_variance"], 0.0)


if __name__ == "__main__":
    unittest.main()
