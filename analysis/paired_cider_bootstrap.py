#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired CIDEr contribution bootstrap for ScanQA and Scan2Cap."""

import argparse
import json
import random
import re
from collections import defaultdict

from pycocoevalcap.cider.cider import Cider


def load_json_or_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        raw = handle.read().strip()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return rows


def percentile(values, q):
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def cider_contributions(targets, predictions):
    keys = sorted(targets)
    ordered_targets = {key: targets[key] for key in keys}
    ordered_predictions = {key: predictions[key] for key in keys}
    aggregate, scores = Cider().compute_score(ordered_targets, ordered_predictions)
    return aggregate, dict(zip(keys, map(float, scores)))


def scanqa_inputs(prediction_path, gt_path):
    predictions = {
        row["question_id"]: row["text"]
        for row in load_json_or_jsonl(prediction_path)
    }
    gt_rows = load_json_or_jsonl(gt_path)
    targets = {}
    hypotheses = {}
    scenes = {}
    for row in gt_rows:
        key = row["question_id"]
        if key not in predictions:
            continue
        targets[key] = [simple_tokenize(text) for text in row["text"]]
        hypotheses[key] = [simple_tokenize(predictions[key])]
        scenes[key] = row.get("scene_id", key.split("-")[1])
    return targets, hypotheses, scenes


def scan2cap_inputs(prediction_path, question_path, corpus_path, threshold):
    prediction_rows = load_json_or_jsonl(prediction_path)
    predictions = {
        (row["scene_id"], int(row["pred_id"]), int(row["gt_id"])): row["text"]
        for row in prediction_rows
    }
    question_rows = load_json_or_jsonl(question_path)
    corpus = load_json_or_jsonl(corpus_path)
    targets = {}
    hypotheses = {}
    scenes = {}
    for key in sorted(corpus):
        targets[key] = list(corpus[key])
        hypotheses[key] = ["sos eos"]
        scenes[key] = key.split("|")[0]
    for row in question_rows:
        pred_key = (row["scene_id"], int(row["pred_id"]), int(row["obj_id"]))
        corpus_key = f"{row['scene_id']}|{row['obj_id']}"
        if pred_key not in predictions or corpus_key not in targets:
            continue
        text = predictions[pred_key] if float(row["iou"]) >= threshold else ""
        hypotheses[corpus_key] = [f"sos {text} eos".replace("\n", " ")]
    return targets, hypotheses, scenes


def bootstrap(deltas, scenes, iterations, seed):
    rng = random.Random(seed)
    keys = sorted(deltas)
    question_estimates = []
    for _ in range(iterations):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        question_estimates.append(sum(deltas[key] for key in sampled) / len(sampled))

    by_scene = defaultdict(list)
    for key in keys:
        by_scene[scenes[key]].append(deltas[key])
    scene_ids = sorted(by_scene)
    scene_estimates = []
    for _ in range(iterations):
        sampled = [scene_ids[rng.randrange(len(scene_ids))] for _ in scene_ids]
        values = [value for scene in sampled for value in by_scene[scene]]
        scene_estimates.append(sum(values) / len(values))
    return (
        (percentile(question_estimates, 0.025), percentile(question_estimates, 0.975)),
        (percentile(scene_estimates, 0.025), percentile(scene_estimates, 0.975)),
        len(keys),
        len(scene_ids),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("scanqa", "scan2cap"), required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gt")
    parser.add_argument("--questions")
    parser.add_argument("--corpus")
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    if args.task == "scanqa":
        if not args.gt:
            parser.error("--gt is required for ScanQA")
        targets, ref_hypotheses, scenes = scanqa_inputs(args.reference, args.gt)
        _, cand_hypotheses, _ = scanqa_inputs(args.candidate, args.gt)
    else:
        if not args.questions or not args.corpus:
            parser.error("--questions and --corpus are required for Scan2Cap")
        targets, ref_hypotheses, scenes = scan2cap_inputs(
            args.reference, args.questions, args.corpus, args.iou_threshold
        )
        _, cand_hypotheses, _ = scan2cap_inputs(
            args.candidate, args.questions, args.corpus, args.iou_threshold
        )

    ref_score, ref_contrib = cider_contributions(targets, ref_hypotheses)
    cand_score, cand_contrib = cider_contributions(targets, cand_hypotheses)
    keys = sorted(set(ref_contrib) & set(cand_contrib))
    deltas = {key: cand_contrib[key] - ref_contrib[key] for key in keys}
    question_ci, scene_ci, n_items, n_scenes = bootstrap(
        deltas, scenes, args.iterations, args.seed
    )
    scale = 100.0
    print(f"n_items={n_items} n_scenes={n_scenes}")
    print(f"reference_cider={ref_score * scale:.4f}")
    print(f"candidate_cider={cand_score * scale:.4f}")
    print(f"delta_pp={(cand_score - ref_score) * scale:.4f}")
    print(
        "question_contribution_bootstrap_95ci_pp="
        f"[{question_ci[0] * scale:.4f}, {question_ci[1] * scale:.4f}]"
    )
    print(
        "scene_contribution_bootstrap_95ci_pp="
        f"[{scene_ci[0] * scale:.4f}, {scene_ci[1] * scale:.4f}]"
    )


if __name__ == "__main__":
    main()
