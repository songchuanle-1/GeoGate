#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired bootstrap for per-query metrics clustered by scene."""

import argparse
import json
import random


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values, q):
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--metric", default="iou")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    def key(row):
        return str(row["scene_id"]), str(row["question_id"])

    reference = {key(row): row for row in load_jsonl(args.reference)}
    candidate = {key(row): row for row in load_jsonl(args.candidate)}
    keys = sorted(set(reference) & set(candidate))
    scene_deltas = {}
    deltas = []
    for item_key in keys:
        delta = (
            float(candidate[item_key][args.metric])
            - float(reference[item_key][args.metric])
        )
        deltas.append(delta)
        scene_deltas.setdefault(item_key[0], []).append(delta)

    scenes = sorted(scene_deltas)
    rng = random.Random(args.seed)
    estimates = []
    for _ in range(args.iterations):
        sampled = [scenes[rng.randrange(len(scenes))] for _ in scenes]
        values = [value for scene in sampled for value in scene_deltas[scene]]
        estimates.append(sum(values) / len(values))

    scale = 100.0
    ci = percentile(estimates, 0.025), percentile(estimates, 0.975)
    ref_mean = sum(float(reference[k][args.metric]) for k in keys) / len(keys)
    cand_mean = sum(float(candidate[k][args.metric]) for k in keys) / len(keys)
    print(f"n_queries={len(keys)} n_scenes={len(scenes)} metric={args.metric}")
    print(f"reference={ref_mean * scale:.4f}")
    print(f"candidate={cand_mean * scale:.4f}")
    print(f"delta_pp={sum(deltas) / len(deltas) * scale:.4f}")
    print(f"scene_bootstrap_95ci_pp=[{ci[0] * scale:.4f}, {ci[1] * scale:.4f}]")


if __name__ == "__main__":
    main()
