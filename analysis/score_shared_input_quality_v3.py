# SPDX-License-Identifier: Apache-2.0
"""Score a complete traced three-condition run; never silently drop pairs."""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.audit_review_v3 import scene_ci, sha256, unique_rows
from analysis.paired_cider_bootstrap import cider_contributions, simple_tokenize
from tools.scanqa_score import clean_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = unique_rows(args.input, ("question_id", "condition"))
    gt = unique_rows(args.gt, ("question_id",))
    if len(records) != 14025 or len(gt) != 4675:
        raise ValueError("Incomplete full-set run")
    keys = sorted(key[0] for key in gt)
    scenes = []
    for key in keys:
        native = records[(key, "native")]
        geo = records[(key, "geogate128")]
        ziprow = records[(key, "zip128")]
        scenes.append(native["scene_id"])
        for field in ("input_tensor_hashes", "prompt_sha256", "seed", "scene_id"):
            if native[field] != geo[field] or native[field] != ziprow[field]:
                raise ValueError(f"Unmatched {field}")
        if native["expanded_input_sha256"] != geo["expanded_input_sha256"]:
            raise ValueError("Different native/GeoGate expanded inputs")
        if geo["actual_kept_spatial_tokens"] != min(128, geo["input_visual_tokens"]):
            raise ValueError("Incorrect GeoGate budget")
        if ziprow["actual_kept_spatial_tokens"] != 128:
            raise ValueError("Unexpected official 3DZip budget")
        for row in (native, geo, ziprow):
            if not row["single_token_decode_verified"]:
                raise ValueError("Unverified decode")
    if len(set(scenes)) != 71:
        raise ValueError("Unexpected scene count")
    targets = {key: [simple_tokenize(clean_answer(t)) for t in gt[(key,)]["text"]]
               for key in keys}
    output = {
        "source_sha256": sha256(args.input), "gt_sha256": sha256(args.gt),
        "n": 4675, "n_scenes": 71,
        "scope": "matched_input_quality; not independent holdout or latency comparison",
        "scorer": "project clean_answer + Java-free tokenization, fixed corpus IDF",
        "seed": 20260908, "iterations": 10000, "conditions": {},
    }
    values = {}
    for condition in ("native", "zip128", "geogate128"):
        predictions = {key: records[(key, condition)]["text"] for key in keys}
        em = np.array([
            int(clean_answer(predictions[key]) in [clean_answer(t) for t in gt[(key,)]["text"]])
            for key in keys
        ]) * 100.0
        aggregate, contributions = cider_contributions(targets, {
            key: [simple_tokenize(clean_answer(predictions[key]))] for key in keys})
        values[condition] = {"EM": em, "CIDEr": np.array([contributions[k] for k in keys]) * 100}
        output["conditions"][condition] = {"EM": float(em.mean()), "CIDEr": aggregate * 100}
    output["comparisons"] = {}
    for first, second in (("native", "geogate128"), ("native", "zip128"),
                          ("zip128", "geogate128")):
        output["comparisons"][f"{second}_minus_{first}"] = {
            metric: scene_ci(values[second][metric] - values[first][metric], scenes)
            for metric in ("EM", "CIDEr")
        }
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
