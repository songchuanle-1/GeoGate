# SPDX-License-Identifier: Apache-2.0
import unittest

from analysis.freeze_cost_controls_v4 import closest_keep, cost


class CostControlsTests(unittest.TestCase):
    def test_matches_exhaustive_nearest(self):
        for spatial in (128, 335, 1497, 2560):
            target = cost(spatial + 220, 348, 14)
            for layer in (2, 8):
                expected = min(range(1, spatial + 1),
                               key=lambda k: (abs(cost(spatial + 220, k + 220, layer)
                                                  - target), k))
                self.assertEqual(closest_keep(spatial, 220, layer, target), expected)

    def test_reference_matches_itself(self):
        self.assertEqual(closest_keep(1497, 220, 14, cost(1717, 348, 14)), 128)
