import unittest

import pandas as pd

from reviewer_experiments.run_211 import rank_candidates


class TwoVariantTests(unittest.TestCase):
    def test_joint_ranking_is_deterministic_and_uses_selection_metrics(self):
        rows = []
        for group in range(7):
            for model in ("logreg", "rf", "hgb"):
                for alpha_step in range(21):
                    rows.append({"feature_set": f"g{group}", "model": model,
                                 "alpha_msnet": alpha_step * 0.05, "threshold": 0.5,
                                 "auc": 0.5, "f1_score": 0.5, "acc": 0.5})
        rows[123]["auc"] = 0.75
        result = rank_candidates(pd.DataFrame(rows))
        self.assertEqual(len(result), 441)
        self.assertEqual(result.iloc[0].feature_set, rows[123]["feature_set"])
        self.assertEqual(result.iloc[0].model, rows[123]["model"])
        self.assertEqual(result.iloc[0].alpha_msnet, rows[123]["alpha_msnet"])

    def test_duplicate_selection_candidate_is_rejected(self):
        row = {"feature_set": "pupil_scanpath", "model": "hgb", "alpha_msnet": 0.8,
               "threshold": 0.5, "auc": 0.9, "f1_score": 0.8, "acc": 0.8}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            rank_candidates(pd.DataFrame([row, row]))

    def test_effectively_equal_auc_uses_f1_tiebreaker(self):
        lower_f1 = {"feature_set": "pupil_scanpath", "model": "hgb", "alpha_msnet": 0.1,
                    "threshold": 0.5, "auc": 0.9750000000000001, "f1_score": 0.92, "acc": 0.9}
        higher_f1 = {"feature_set": "all_behavior", "model": "rf", "alpha_msnet": 0.1,
                     "threshold": 0.5, "auc": 0.975, "f1_score": 0.97, "acc": 0.95}
        self.assertEqual(rank_candidates(pd.DataFrame([lower_f1, higher_f1])).iloc[0].model, "rf")


if __name__ == "__main__":
    unittest.main()
