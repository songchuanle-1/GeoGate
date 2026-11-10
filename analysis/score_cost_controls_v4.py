# SPDX-License-Identifier: Apache-2.0
"""Strict full-set cost-control audit and prespecified paired scene intervals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.freeze_cost_controls_v4 import cost
from analysis.review_v3_protocol import sample_seed

CONDITIONS = ("native", "L2_cost", "L8_cost", "L14_128")


def distribution(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid distribution values")
    return {"min": float(values.min()), "mean": float(values.mean()),
            "max": float(values.max()),
            "p05_p50_p95": np.quantile(values, [0.05, 0.5, 0.95]).tolist()}


def scene_interval(values, scenes, level=0.975, iterations=10000):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) != len(scenes) or not len(values)
            or not np.isfinite(values).all() or not 0 < level < 1
            or not isinstance(iterations, int) or iterations < 1):
        raise ValueError("Invalid paired values")
    _, labels = np.unique(scenes, return_inverse=True)
    counts = np.bincount(labels)
    sums = np.bincount(labels, weights=values)
    draws = np.random.default_rng(20260910).integers(
        0, len(counts), (iterations, len(counts)))
    estimates = sums[draws].sum(1) / counts[draws].sum(1)
    alpha = (1 - level) / 2
    return {"mean": float(values.mean()), "level": level,
            "ci": np.quantile(estimates, [alpha, 1 - alpha]).tolist()}


def audit_group(rows, budget):
    if set(rows) != set(CONDITIONS):
        raise ValueError("Missing or extra conditions")
    reference = rows["native"]
    spatial, text = budget["spatial"], budget["text"]
    full = spatial + text
    for condition, row in rows.items():
        for field in ("question_id", "scene_id", "seed", "prompt",
                      "input_tensor_hashes", "expanded_input_sha256"):
            if row[field] != reference[field]:
                raise ValueError(f"Unmatched {field}")
        if row["seed"] != sample_seed(row["question_id"], 20260908):
            raise ValueError("Incorrect seed")
        layer = {"native": None, "L2_cost": 2, "L8_cost": 8, "L14_128": 14}[condition]
        keep = (spatial if layer is None else 128 if layer == 14 else
                budget["controls"][str(layer)]["keep"])
        reduced = text + keep
        lengths = ([full] * 32 if layer is None else
                   [full] * (layer + 1) + [reduced] * (31 - layer))
        if (row["prune_layer"] != layer or row["input_visual_tokens"] != spatial
                or row["actual_kept_spatial_tokens"] != keep
                or row["prefill_layer_input_lengths"] != lengths
                or row["prefill_kv_lengths"] != [reduced] * 32
                or row["single_token_decode_verified"] is not True):
            raise ValueError("Invalid token or cache trace")
        indices = row["selected_original_indices"]
        if (indices != sorted(set(indices)) or len(indices) != reduced
                or any(i < 0 or i >= full for i in indices)):
            raise ValueError("Invalid selected indices")
        if layer in (2, 8):
            ratio = cost(full, reduced, layer) / cost(full, text + 128, 14)
            if abs(ratio - 1) > 0.002:
                raise ValueError("Cost mismatch exceeds frozen rounding envelope")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from analysis.paired_cider_bootstrap import cider_contributions, simple_tokenize
    from tools.scanqa_score import clean_answer
    budgets_list = json.loads(args.budgets.read_text())["samples"]
    budgets = {str(r["question_id"]): r for r in budgets_list}
    gt_list = json.loads(args.gt.read_text())
    gt = {str(r["question_id"]): r for r in gt_list}
    question_list = json.loads(args.questions.read_text())
    questions = {str(r["question_id"]): r for r in question_list}
    if (len(budgets_list) != len(budgets) or len(gt_list) != len(gt)
            or len(question_list) != len(questions) or set(questions) != set(gt)
            or len(gt) != 4675 or set(gt) != set(budgets)):
        raise ValueError("Ground truth and budget key sets mismatch")
    records = {}
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["question_id"]), row["condition"])
            if key in records:
                raise ValueError("Duplicate output")
            records[key] = row
    if set(records) != {(q, c) for q in gt for c in CONDITIONS}:
        raise ValueError("Incomplete full set")
    keys = sorted(gt)
    scenes = []
    for q in keys:
        audit_group({c: records[q, c] for c in CONDITIONS}, budgets[q])
        if records[q, "native"]["scene_id"] != questions[q]["video"]:
            raise ValueError("Output scene differs from frozen ground truth")
        scenes.append(records[q, "native"]["scene_id"])
    if len(set(scenes)) != 71:
        raise ValueError("Unexpected scene set")
    targets = {q: [simple_tokenize(clean_answer(t)) for t in gt[q]["text"]] for q in keys}
    output = {"n": len(keys), "n_scenes": 71, "seed": 20260910,
              "iterations": 10000, "conditions": {}, "comparisons": {},
              "primary_endpoints": ["L14_128_minus_L2_cost:EM",
                                    "L14_128_minus_L8_cost:EM"],
              "multiplicity": "two prespecified EM contrasts, Bonferroni 97.5% intervals",
              "hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in (("input", args.input), ("budgets", args.budgets),
                                            ("gt", args.gt), ("questions", args.questions),
                                            ("scorer", Path(__file__)))},
              "scope": "analytic LM cost matched; not latency or independent holdout"}
    values = {}
    for c in CONDITIONS:
        predictions = {q: clean_answer(records[q, c]["text"]) for q in keys}
        em = np.array([100 * int(predictions[q] in [clean_answer(t) for t in gt[q]["text"]])
                       for q in keys])
        mean, contrib = cider_contributions(
            targets, {q: [simple_tokenize(predictions[q])] for q in keys})
        values[c] = {"EM": em, "CIDEr": np.array([contrib[q] * 100 for q in keys])}
        output["conditions"][c] = {
            "EM": float(em.mean()), "CIDEr": mean * 100,
            "retained_spatial_tokens": distribution([
                records[q, c]["actual_kept_spatial_tokens"] for q in keys]),
        }
        if c in ("L2_cost", "L8_cost"):
            layer = 2 if c == "L2_cost" else 8
            output["conditions"][c]["relative_analytic_cost_error"] = distribution([
                cost(budgets[q]["spatial"] + budgets[q]["text"],
                     records[q, c]["actual_kept_spatial_tokens"] + budgets[q]["text"],
                     layer) / cost(budgets[q]["spatial"] + budgets[q]["text"],
                                   128 + budgets[q]["text"], 14) - 1
                for q in keys])
    for control in ("L2_cost", "L8_cost", "native"):
        output["comparisons"][f"L14_128_minus_{control}"] = {
            metric: scene_interval(values["L14_128"][metric] - values[control][metric],
                                   scenes, level=0.975 if metric == "EM" and
                                   control != "native" else 0.95)
            for metric in ("EM", "CIDEr")}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
