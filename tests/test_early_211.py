import unittest

import torch
from torch import nn


class EarlyFusionTests(unittest.TestCase):
    def test_extracts_head_input_instead_of_probability(self):
        from reviewer_experiments.early_211 import extract_embedding

        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Sequential(nn.Flatten(), nn.Linear(56, 1), nn.Sigmoid())

            def forward(self, x, alpha=1.0):
                return self.head(x)

        model = SmallModel()
        source = torch.arange(112, dtype=torch.float32).reshape(2, 1, 56)
        extracted = extract_embedding(model, source)
        self.assertEqual(tuple(extracted.shape), (2, 56))
        torch.testing.assert_close(extracted, source.flatten(1))

    def test_new_head_accepts_229_features(self):
        from reviewer_experiments.early_211 import make_head

        head = make_head()
        self.assertEqual(tuple(head(torch.zeros(3, 229)).shape), (3, 1))
        self.assertEqual(head[0].in_features, 229)


if __name__ == "__main__":
    unittest.main()
