import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from reviewer_experiments.select_211 import choose_configuration, select_inside_training


class Selection211Tests(unittest.TestCase):
    def test_inner_selection_never_reads_alpha_or_evaluation_people(self):
        frames = []
        for fold_number in range(4):
            for row in range(40):
                frames.append({"subject_int": fold_number * 40 + row,
                               "set_name": f"Set_{fold_number}", "label": row % 2,
                               "value": row / 40})
        data = pd.DataFrame(frames)
        feature_sets = {f"group_{i}": ["value"] for i in range(7)}
        fitted = []

        class FakeModel:
            def fit(self, x, y):
                fitted.append(len(x))
                return self

            def predict_proba(self, x):
                p = np.clip(x.value.to_numpy(), 0.01, 0.99)
                return np.column_stack([1 - p, p])

        with patch("reviewer_experiments.select_211.make_behavior_model", return_value=FakeModel()):
            first = select_inside_training(data, feature_sets, ("Set_2", "Set_3"), 42)
            changed = data.copy()
            changed.loc[changed.set_name.isin(["Set_0", "Set_1"]), "label"] ^= 1
            second = select_inside_training(changed, feature_sets, ("Set_2", "Set_3"), 42)
        self.assertEqual(fitted, [40] * 84)
        self.assertEqual(len(first[0]), 42)
        self.assertEqual(len(first[1]), 21)
        pd.testing.assert_frame_equal(first[1].reset_index(drop=True), second[1].reset_index(drop=True))

    def test_ranking_uses_auc_then_f1_then_accuracy(self):
        rows = [{"feature_set": f"g_{i}", "model": f"m_{i}",
                 "auc": 0.5, "f1_score": 0.5, "acc": 0.5} for i in range(21)]
        rows[7]["auc"] = 0.6
        winner = choose_configuration(pd.DataFrame(rows))
        self.assertEqual(winner.feature_set, "g_7")


if __name__ == "__main__":
    unittest.main()
