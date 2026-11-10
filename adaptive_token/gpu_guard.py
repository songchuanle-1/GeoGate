# SPDX-License-Identifier: Apache-2.0
"""Small dependency-free helpers for refusing shared GPU execution."""

from __future__ import annotations

import os
import subprocess


def compute_pids(gpu_id: int) -> set[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(int(gpu_id)),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        int(line.strip())
        for line in output.splitlines()
        if line.strip()
    }


def guard_gpu(
    gpu_id: int,
    allowed_pids: set[int],
    context: str,
) -> None:
    foreign = compute_pids(gpu_id) - set(allowed_pids)
    if foreign:
        raise RuntimeError(
            f"{context}: foreign GPU PIDs on device {gpu_id}: "
            f"{sorted(foreign)}"
        )


def gpu_id_from_environment() -> int:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not value:
        raise RuntimeError(
            "--gpu-id is required when CUDA_VISIBLE_DEVICES is unset"
        )
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError(
            "GPU guard requires exactly one visible CUDA device"
        )
    try:
        return int(devices[0])
    except ValueError as error:
        raise RuntimeError(
            f"Unsupported CUDA_VISIBLE_DEVICES value: {value!r}"
        ) from error
