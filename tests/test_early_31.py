import unittest

import numpy as np
import pandas as pd


class Early31Tests(unittest.TestCase):
    def test_outer_validation_never_enters_head_training_or_epoch_selection(self):
        from reviewer_experiments.early_31 import fold_roles_31

        folds = {f"Set_{i}" for i in range(4)}
        for evaluation in folds:
            train, selection_train, selection = fold_roles_31(evaluation)
            self.assertEqual(set(train), folds - {evaluation})
            self.assertEqual(set(selection_train), folds - {evaluation, selection})
            self.assertNotIn(evaluation, selection_train)
            self.assertNotEqual(evaluation, selection)

    def test_checkpoint_parity_requires_same_subject_scores_and_labels(self):
        from reviewer_experiments.early_31 import verify_pretrained_parity

        expected = pd.DataFrame({
            "subject_int": [2, 8], "set_name": ["Set_0", "Set_0"],
            "label": [0, 1], "msnet_prob": [0.2, 0.8],
        })
        observed = pd.DataFrame({
            "subject_int": [8, 2], "set_name": ["Set_0", "Set_0"],
            "label": [1, 0], "msnet_prob": [0.8, 0.2],
        })
        self.assertLess(verify_pretrained_parity(observed, expected, "Set_0"), 1e-12)
        observed.loc[0, "msnet_prob"] = 0.7
        with self.assertRaisesRegex(ValueError, "MSNet probability mismatch"):
            verify_pretrained_parity(observed, expected, "Set_0")
        observed.loc[0, "msnet_prob"] = 0.8
        observed.loc[0, "label"] = 0
        with self.assertRaisesRegex(ValueError, "subject/fold/label"):
            verify_pretrained_parity(observed, expected, "Set_0")

    def test_preprocessing_is_fit_only_on_named_training_rows(self):
        from reviewer_experiments.early_31 import prepare_combined

        embeddings = np.zeros((4, 56), dtype=np.float32)
        behavioral = pd.DataFrame(np.zeros((4, 173)), columns=[f"x{i}" for i in range(173)])
        behavioral.loc[:, "x0"] = [0, 2, 100, 200]
        combined, imputer, scaler = prepare_combined(
            embeddings, behavioral, list(behavioral.columns), np.array([0, 1]),
        )
        self.assertEqual(combined.shape, (4, 229))
        self.assertAlmostEqual(float(imputer.statistics_[0]), 1.0)
        self.assertAlmostEqual(float(scaler.mean_[0]), 1.0)
        self.assertGreater(float(combined[2, 56]), 50.0)


if __name__ == "__main__":
    unittest.main()
