#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Clustered bootstrap audit of the three-phase attention structure."""

import argparse
import json
import random
from collections import Counter, defaultdict


def percentile(values, q):
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        rows = json.load(handle)["per_sample"]

    by_scene = defaultdict(list)
    records = []
    for row in rows:
        frac = row["frac"]
        top50 = row["top50"]
        record = {
            "phase_delta": mean(frac[14:27]) - mean(frac[2:14]),
            "boundary_delta": frac[14] - frac[13],
            "coverage_delta": mean(top50[14:27]) - mean(top50[2:14]),
            "peak_layer": 9 + max(range(18), key=lambda i: frac[9 + i]),
        }
        records.append(record)
        scene_id = row.get("video", row.get("scene_id"))
        if scene_id is None:
            raise KeyError("per-sample row requires video or scene_id")
        by_scene[scene_id].append(record)

    scenes = sorted(by_scene)
    rng = random.Random(args.seed)
    boot = {key: [] for key in ("phase_delta", "boundary_delta", "coverage_delta")}
    for _ in range(args.iterations):
        sampled = [scenes[rng.randrange(len(scenes))] for _ in scenes]
        sample_rows = [record for scene in sampled for record in by_scene[scene]]
        for key in boot:
            boot[key].append(mean([record[key] for record in sample_rows]))

    result = {
        "n_questions": len(records),
        "n_scenes": len(scenes),
        "phase_delta": mean([record["phase_delta"] for record in records]),
        "phase_delta_scene_bootstrap_95ci": [
            percentile(boot["phase_delta"], 0.025),
            percentile(boot["phase_delta"], 0.975),
        ],
        "phase_delta_positive_fraction": mean(
            [record["phase_delta"] > 0 for record in records]
        ),
        "boundary_delta_l14_minus_l13": mean(
            [record["boundary_delta"] for record in records]
        ),
        "boundary_delta_scene_bootstrap_95ci": [
            percentile(boot["boundary_delta"], 0.025),
            percentile(boot["boundary_delta"], 0.975),
        ],
        "boundary_delta_positive_fraction": mean(
            [record["boundary_delta"] > 0 for record in records]
        ),
        "coverage_delta": mean([record["coverage_delta"] for record in records]),
        "coverage_delta_scene_bootstrap_95ci": [
            percentile(boot["coverage_delta"], 0.025),
            percentile(boot["coverage_delta"], 0.975),
        ],
        "late_peak_histogram": dict(sorted(Counter(
            record["peak_layer"] for record in records
        ).items())),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
