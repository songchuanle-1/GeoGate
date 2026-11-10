# SPDX-License-Identifier: Apache-2.0
import unittest

try:
    import torch
    from adaptive_token.coverage_selector import (
        keep_after_topk_deletion,
        select_by_score_and_coverage,
    )
except ModuleNotFoundError:
    torch = None
    keep_after_topk_deletion = None
    select_by_score_and_coverage = None


@unittest.skipIf(torch is None, "PyTorch is required for coverage selector tests")
class CoverageSelectorTest(unittest.TestCase):
    def test_first_choice_is_highest_score(self):
        scores = torch.tensor([0.1, 0.9, 0.8, 0.2])
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        selected = select_by_score_and_coverage(scores, coords, 2, 0.25)
        self.assertEqual(selected[0].item(), 1)
        self.assertEqual(len(set(selected.tolist())), 2)

    def test_missing_coordinates_falls_back_to_topk(self):
        scores = torch.tensor([0.1, 0.9, 0.8, 0.2])
        selected = select_by_score_and_coverage(scores, None, 2)
        self.assertEqual(set(selected.tolist()), {1, 2})

    def test_invalid_k_is_clamped(self):
        scores = torch.tensor([0.1, 0.9])
        coords = torch.zeros(2, 3)
        selected = select_by_score_and_coverage(scores, coords, 10)
        self.assertEqual(selected.numel(), 2)

    def test_delete_top_scores(self):
        scores = torch.tensor([0.1, 0.9, 0.8, 0.2])
        kept = keep_after_topk_deletion(scores, 2)
        self.assertEqual(kept.tolist(), [0, 3])

    def test_deletion_never_removes_every_token(self):
        scores = torch.tensor([0.1, 0.9])
        kept = keep_after_topk_deletion(scores, 10)
        self.assertEqual(kept.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
