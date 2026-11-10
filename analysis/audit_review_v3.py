# SPDX-License-Identifier: Apache-2.0
"""Strict artifact pairing and scene bootstrap for the 2026-09-08 follow-up."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.paired_scanqa_bootstrap import exact_match, load_json_or_jsonl


def unique_rows(path, fields):
    rows = load_json_or_jsonl(path)
    keyed = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if key in keyed:
            raise ValueError(f"Duplicate key in {path}: {key}")
        keyed[key] = row
    return keyed


def paired_keys(reference, candidate, expected):
    if set(reference) != set(candidate):
        raise ValueError("Pairing requires identical key sets, not their intersection")
    if len(reference) != expected:
        raise ValueError(f"Expected {expected} pairs, found {len(reference)}")
    return sorted(reference)


def scene_ci(values, scenes, iterations=10000, seed=20260908):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Values must be nonempty and finite")
    scene_ids, labels = np.unique(scenes, return_inverse=True)
    sums = np.bincount(labels, weights=values)
    counts = np.bincount(labels)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(scene_ids), size=(iterations, len(scene_ids)))
    estimates = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "n": len(values), "n_scenes": len(scene_ids),
        "mean": float(values.mean()),
        "scene_ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
    }


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid distribution")
    return {
        "n": len(values), "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    }


def compare_metric(reference, candidate, keys, metric, group=None):
    if group is not None:
        keys = [k for k in keys if group(reference[k])]
    if not keys:
        return {"n": 0, "status": "empty"}
    a = np.array([reference[k][metric] for k in keys], dtype=float) * 100
    b = np.array([candidate[k][metric] for k in keys], dtype=float) * 100
    if np.any((a < 0) | (a > 100) | (b < 0) | (b > 100)):
        raise ValueError(f"{metric} outside [0, 1]")
    return {
        "reference": float(a.mean()), "candidate": float(b.mean()),
        "delta": scene_ci(b - a, [k[0] for k in keys]),
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    retrieved = root / "results/raw/followup/review_v3_20260908"
    zipdir = root / "results/raw/followup/3dzip_v1"
    paths = {
        "multi_base": retrieved / "multi3drefer_baseline.jsonl",
        "multi_r10": retrieved / "multi3drefer_geogate_L14_R010.jsonl",
        "multi_r20": retrieved / "multi3drefer_geogate_L14_R020.jsonl",
        "native": zipdir / "llava3d_native_officialrunner_full.jsonl",
        "zip128": zipdir / "llava3d_3dzip_t128_official_full.jsonl",
        "geogate128": retrieved / "llava3d_geogate_t128_officialstack_full.jsonl",
        "gt": root / "third_party/LLaVA-3D/playground/data/annotations/llava3d_scanqa_val_answer.json",
    }
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": 20260908, "iterations": 10000,
        "sources": {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()},
        "status": "artifact_scoring_not_historical_pruning_provenance_validation",
        "latency_scope": "descriptive_only_unmatched_run_order_and_timing_boundaries",
    }
    multi = {k: unique_rows(paths[k], ("scene_id", "question_id"))
             for k in ("multi_base", "multi_r10", "multi_r20")}
    reference = multi["multi_base"]
    output["multi3drefer"] = {}
    groups = {
        "all": None,
        "zero": lambda row: len(row["object_ids"]) == 0,
        "one": lambda row: len(row["object_ids"]) == 1,
        "multiple": lambda row: len(row["object_ids"]) > 1,
    }
    for name in ("multi_r10", "multi_r20"):
        candidate = multi[name]
        keys = paired_keys(reference, candidate, 11120)
        for key in keys:
            for field in ("prompt", "object_ids", "model_id"):
                if reference[key][field] != candidate[key][field]:
                    raise ValueError(f"{name} mismatched {field}: {key}")
        output["multi3drefer"][name] = {
            group: compare_metric(reference, candidate, keys, "iou", select)
            for group, select in groups.items()
        }
    keys = paired_keys(multi["multi_r10"], multi["multi_r20"], 11120)
    output["multi3drefer"]["r20_minus_r10"] = {
        group: compare_metric(multi["multi_r10"], multi["multi_r20"], keys, "iou", select)
        for group, select in groups.items()
    }
    qa = {k: unique_rows(paths[k], ("question_id",))
          for k in ("native", "zip128", "geogate128")}
    gt = unique_rows(paths["gt"], ("question_id",))
    keys = paired_keys(qa["native"], gt, 4675)
    for name in ("zip128", "geogate128"):
        paired_keys(qa["native"], qa[name], 4675)
        for key in keys:
            if qa["native"][key]["prompt"] != qa[name][key]["prompt"]:
                raise ValueError(f"{name} prompt mismatch: {key}")
    scenes = [qa["geogate128"][k]["scene_id"] for k in keys]
    token_mismatch = [
        k for k in keys if qa["native"][k]["visual_tokens"]
        != qa["geogate128"][k]["input_visual_tokens"]
    ]
    output["qa_pairing"] = {
        "n": len(keys), "n_scenes": len(set(scenes)),
        "prompt_mismatches": 0,
        "native_logged_precap_vs_geogate_postcap_count_differences": len(token_mismatch),
        "native_2560_cap_adjusted_count_mismatches": sum(
            min(2560, qa["native"][k]["visual_tokens"])
            != qa["geogate128"][k]["input_visual_tokens"] for k in keys),
        "count_semantics": "native logs precap tower voxels; GeoGate logs postcap LLM input",
        "input_tensor_hashes_recorded": False,
        "actual_geogate_postprune_counts_recorded": False,
        "geogate_target_histogram": dict(Counter(
            qa["geogate128"][k]["kept_visual_tokens"] for k in keys)),
        "geogate_layer_histogram": dict(Counter(
            qa["geogate128"][k]["prune_layer"] for k in keys)),
    }
    from analysis.paired_cider_bootstrap import cider_contributions, simple_tokenize
    from tools.scanqa_score import clean_answer

    scores, output["scanqa"] = {}, {}
    # Preserve the old audit protocol and report the project scorer separately.
    for protocol in ("legacy_audit", "project_scorer"):
        normalize = (lambda text: text) if protocol == "legacy_audit" else clean_answer
        targets = {k[0]: [simple_tokenize(normalize(t)) for t in gt[k]["text"]]
                   for k in keys}
        scores[protocol] = {}
        for name, rows in qa.items():
            em = np.array([
                exact_match(rows[k]["text"], gt[k]["text"])
                if protocol == "legacy_audit" else
                int(clean_answer(rows[k]["text"]) in [clean_answer(t) for t in gt[k]["text"]])
                for k in keys
            ]) * 100.0
            aggregate, contributions = cider_contributions(targets, {
                k[0]: [simple_tokenize(normalize(rows[k]["text"]))] for k in keys
            })
            scores[protocol][name] = {
                "EM": em, "CIDEr": np.array([contributions[k[0]] for k in keys]) * 100,
            }
            output["scanqa"].setdefault(protocol, {})[name] = {
                "EM": float(em.mean()), "CIDEr": aggregate * 100,
            }
        comparisons = {}
        for first, second in (("native", "zip128"), ("native", "geogate128"),
                              ("zip128", "geogate128")):
            comparisons[f"{second}_minus_{first}"] = {
                metric: scene_ci(scores[protocol][second][metric] -
                                 scores[protocol][first][metric], scenes)
                for metric in ("EM", "CIDEr")
            }
        output["scanqa"][protocol]["comparisons"] = comparisons
    output["latency_descriptive_seconds"] = {
        name: distribution([row["latency_s"] for row in rows.values()])
        for name, rows in {**qa, **multi}.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({k: output[k] for k in ("qa_pairing", "multi3drefer", "scanqa")}, indent=2))


if __name__ == "__main__":
    main()
