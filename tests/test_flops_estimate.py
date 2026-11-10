# SPDX-License-Identifier: Apache-2.0
import unittest

from analysis.flops_estimate import estimate


class FlopsEstimateTest(unittest.TestCase):
    def test_prune_layer_runs_before_reduction(self):
        stats = estimate(1497, 220, 14, 0.10, decode_steps=8)
        self.assertEqual(stats["full_blocks"], 15)
        self.assertEqual(stats["reduced_blocks"], 17)
        self.assertAlmostEqual(stats["total_ratio"], 0.5791172, places=6)
        self.assertAlmostEqual(stats["kv_ratio"], 0.2154921, places=6)

    def test_early_pruning_uses_three_full_blocks_at_l2(self):
        stats = estimate(1497, 220, 2, 0.50, decode_steps=8)
        self.assertEqual(stats["full_blocks"], 3)
        self.assertEqual(stats["reduced_blocks"], 29)
        self.assertAlmostEqual(stats["total_ratio"], 0.5919554, places=6)


if __name__ == "__main__":
    unittest.main()
