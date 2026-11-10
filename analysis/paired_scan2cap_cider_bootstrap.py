#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired Scan2Cap CIDEr bootstrap using the official evaluation inputs."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values, q):
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def prepare(prediction_path, pred_attrs, gt_attrs, corpus_path, threshold, box_utils):
    predictions = load_jsonl(prediction_path)
    instance_attrs = torch.load(pred_attrs, map_location="cpu")
    scannet_attrs = torch.load(gt_attrs, map_location="cpu")
    corpus = json.load(open(corpus_path, encoding="utf-8"))

    hypotheses = {
        key: [{"caption": "sos eos"}]
        for key in corpus
    }
    targets = {
        key: [{"caption": caption} for caption in captions]
        for key, captions in corpus.items()
    }
    for row in predictions:
        scene_id = row["scene_id"]
        pred_id = row["pred_id"]
        gt_id = row["gt_id"]
        pred_locs = instance_attrs[scene_id]["locs"][pred_id].tolist()
        gt_locs = scannet_attrs[scene_id]["locs"][gt_id].tolist()
        pred_corners = box_utils.construct_bbox_corners(pred_locs[:3], pred_locs[3:])
        gt_corners = box_utils.construct_bbox_corners(gt_locs[:3], gt_locs[3:])
        iou = box_utils.box3d_iou(pred_corners, gt_corners)
        key = f"{scene_id}|{gt_id}"
        text = row["text"] if iou >= threshold else ""
        hypotheses[key] = [{"caption": f"sos {text} eos".replace("\n", " ")}]

    tokenizer = PTBTokenizer()
    return tokenizer.tokenize(targets), tokenizer.tokenize(hypotheses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--pred-attrs", required=True)
    parser.add_argument("--gt-attrs", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.eval_dir).resolve()))
    import box_utils

    targets, ref_hypotheses = prepare(
        args.reference, args.pred_attrs, args.gt_attrs, args.corpus,
        args.iou_threshold, box_utils,
    )
    _, cand_hypotheses = prepare(
        args.candidate, args.pred_attrs, args.gt_attrs, args.corpus,
        args.iou_threshold, box_utils,
    )

    keys = sorted(targets)
    targets = {key: targets[key] for key in keys}
    ref_hypotheses = {key: ref_hypotheses[key] for key in keys}
    cand_hypotheses = {key: cand_hypotheses[key] for key in keys}
    ref_score, ref_scores = Cider().compute_score(targets, ref_hypotheses)
    cand_score, cand_scores = Cider().compute_score(targets, cand_hypotheses)
    deltas = {
        key: float(cand_scores[index] - ref_scores[index])
        for index, key in enumerate(keys)
    }

    by_scene = defaultdict(list)
    for key in keys:
        by_scene[key.split("|")[0]].append(deltas[key])
    scenes = sorted(by_scene)
    rng = random.Random(args.seed)
    estimates = []
    for _ in range(args.iterations):
        sampled = [scenes[rng.randrange(len(scenes))] for _ in scenes]
        values = [value for scene in sampled for value in by_scene[scene]]
        estimates.append(sum(values) / len(values))

    scale = 100.0
    print(f"n_items={len(keys)} n_scenes={len(scenes)}")
    print(f"reference_cider={ref_score * scale:.4f}")
    print(f"candidate_cider={cand_score * scale:.4f}")
    print(f"delta_pp={(cand_score - ref_score) * scale:.4f}")
    print(
        "scene_contribution_bootstrap_95ci_pp="
        f"[{percentile(estimates, 0.025) * scale:.4f}, "
        f"{percentile(estimates, 0.975) * scale:.4f}]"
    )


if __name__ == "__main__":
    main()
