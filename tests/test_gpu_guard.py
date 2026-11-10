# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest.mock import patch

from adaptive_token.gpu_guard import (
    compute_pids,
    gpu_id_from_environment,
    guard_gpu,
)


class GpuGuardTests(unittest.TestCase):
    def test_compute_pids_parses_nvidia_smi_output(self):
        with patch(
            "adaptive_token.gpu_guard.subprocess.check_output",
            return_value="123\n456\n\n",
        ):
            self.assertEqual(compute_pids(3), {123, 456})

    def test_guard_rejects_foreign_process(self):
        with patch(
            "adaptive_token.gpu_guard.compute_pids",
            return_value={10, 11},
        ):
            with self.assertRaisesRegex(RuntimeError, "foreign GPU PIDs"):
                guard_gpu(2, {10}, "question 4")

    def test_guard_accepts_allowed_processes(self):
        with patch(
            "adaptive_token.gpu_guard.compute_pids",
            return_value={10},
        ):
            guard_gpu(2, {10}, "question 4")

    def test_gpu_id_from_environment_requires_single_device(self):
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "5"},
            clear=False,
        ):
            self.assertEqual(gpu_id_from_environment(), 5)
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "5,6"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                gpu_id_from_environment()


if __name__ == "__main__":
    unittest.main()
