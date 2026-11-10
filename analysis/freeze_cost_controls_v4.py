# SPDX-License-Identifier: Apache-2.0
"""Choose single-drop cost controls from sequence lengths, never predictions."""

import argparse
import hashlib
import json
from pathlib import Path

from analysis.flops_estimate import layer_decode, layer_prefill


def cost(full, reduced, layer, steps=8):
    return ((layer + 1) * layer_prefill(full)
            + (31 - layer) * layer_prefill(reduced)
            + steps * 32 * layer_decode(reduced))


def closest_keep(spatial, text, layer, target):
    low, high = 1, spatial
    while low < high:
        mid = (low + high) // 2
        if cost(spatial + text, mid + text, layer) < target:
            low = mid + 1
        else:
            high = mid
    return min({low, max(1, low - 1)},
               key=lambda k: (abs(cost(spatial + text, k + text, layer) - target), k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = []
    seen = set()
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] != "native":
                continue
            qid = str(row["question_id"])
            if qid in seen:
                raise ValueError("Duplicate native question")
            seen.add(qid)
            spatial = row["input_visual_tokens"]
            full = row["prefill_layer_input_lengths"][0]
            text = full - spatial
            if not 128 <= spatial <= full:
                raise ValueError("Invalid sequence lengths")
            target = cost(full, text + 128, 14)
            controls = {}
            for layer in (2, 8):
                keep = closest_keep(spatial, text, layer, target)
                controls[str(layer)] = {
                    "keep": keep,
                    "relative_cost_error": cost(full, text + keep, layer) / target - 1,
                }
            samples.append({"question_id": qid, "spatial": spatial,
                            "text": text, "controls": controls})
    if len(samples) != 4675:
        raise ValueError("Expected the full 4675-question length manifest")
    output = {
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "scope": "LM analytic cost controls only; not measured latency or full-model FLOPs",
        "target": {"layer": 14, "keep": 128, "assumed_decode_steps": 8},
        "selection": "Uses native lengths only; no answers or quality scores",
        "n": len(samples), "summary": {}, "samples": samples,
    }
    for layer in ("2", "8"):
        controls = [r["controls"][layer] for r in samples]
        output["summary"][layer] = {
            "mean_keep": sum(r["keep"] for r in controls) / len(controls),
            "min_keep": min(r["keep"] for r in controls),
            "max_keep": max(r["keep"] for r in controls),
            "max_abs_relative_error": max(abs(r["relative_cost_error"]) for r in controls),
        }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
