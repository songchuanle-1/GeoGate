# SPDX-License-Identifier: Apache-2.0
import unittest

try:
    import torch
    from adaptive_token.geogate_v0 import snapshot_indices
except ModuleNotFoundError:
    torch = None
    snapshot_indices = None


@unittest.skipIf(torch is None, "PyTorch is required for KV mapping tests")
class KvSnapshotMappingTest(unittest.TestCase):
    def test_maps_final_tokens_into_full_snapshot(self):
        snapshot = torch.tensor([0, 1, 2, 4, 7, 9])
        final_map = torch.tensor([0, 2, 7, 9])
        self.assertEqual(
            snapshot_indices(snapshot, final_map).tolist(),
            [0, 2, 4, 5],
        )

    def test_rejects_missing_final_token(self):
        snapshot = torch.tensor([0, 1, 4, 7])
        final_map = torch.tensor([0, 2, 7])
        with self.assertRaises(ValueError):
            snapshot_indices(snapshot, final_map)


if __name__ == "__main__":
    unittest.main()
