# SPDX-License-Identifier: Apache-2.0
import unittest

try:
    import torch
    from adaptive_token.geo_coords import pool_voxel_coordinates
    from adaptive_token.coverage_selector import merge_spatial_selection
except ModuleNotFoundError:
    torch = None
    pool_voxel_coordinates = None
    merge_spatial_selection = None


@unittest.skipIf(torch is None, "PyTorch and torch_scatter are required")
class GeoCoordinateMappingTest(unittest.TestCase):
    def test_pools_coordinates_per_batch_in_voxel_order(self):
        feat_xyz = torch.tensor([
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
               [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]],
            [[[[10.0, 0.0, 0.0], [10.0, 0.0, 0.0],
               [12.0, 0.0, 0.0], [12.0, 0.0, 0.0]]]],
        ])
        p2v = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ])
        pooled = pool_voxel_coordinates(feat_xyz, p2v)
        expected = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ])
        self.assertTrue(torch.equal(pooled, expected))

    def test_rejects_mismatched_point_count(self):
        feat_xyz = torch.zeros(1, 1, 1, 4, 3)
        with self.assertRaises(ValueError):
            pool_voxel_coordinates(feat_xyz, torch.zeros(1, 3, dtype=torch.long))

    def test_spatial_merge_preserves_nonspatial_order(self):
        selected = torch.tensor([2, 0], dtype=torch.long)
        merged = merge_spatial_selection(2, 5, 8, selected)
        self.assertEqual(merged.tolist(), [0, 1, 2, 4, 5, 6, 7])
        original_positions = torch.arange(8)
        self.assertEqual(
            original_positions[merged].tolist(),
            [0, 1, 2, 4, 5, 6, 7],
        )


if __name__ == "__main__":
    unittest.main()
