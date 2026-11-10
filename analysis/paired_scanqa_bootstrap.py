#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired ScanQA EM analysis with question- and scene-level bootstrap."""

import argparse
import json
import random
import re
from collections import defaultdict


NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
]


def clean_answer(text):
    text = text.lower().strip()
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", "", text)
    for digit, word in enumerate(NUMBER_WORDS):
        text = re.sub(rf"\b{digit}\b", word, text)
    return text


def load_json_or_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        raw = handle.read().strip()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = rows.get("answers", rows.get("data", [rows]))
    return rows


def exact_match(prediction, references):
    if isinstance(references, str):
        references = [references]
    pred = clean_answer(prediction)
    return int(any(pred == clean_answer(ref) for ref in references))


def percentile(values, q):
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def bootstrap_question(deltas, iterations, seed):
    rng = random.Random(seed)
    n = len(deltas)
    estimates = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def bootstrap_scene(scene_deltas, iterations, seed):
    rng = random.Random(seed)
    scenes = list(scene_deltas)
    estimates = []
    for _ in range(iterations):
        sample = [scenes[rng.randrange(len(scenes))] for _ in scenes]
        values = [delta for scene in sample for delta in scene_deltas[scene]]
        estimates.append(sum(values) / len(values))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument(
        "--question-meta",
        help="Optional JSON/JSONL questions providing scene_id/video metadata.",
    )
    parser.add_argument(
        "--exclude",
        help="Optional JSON/JSONL file whose question_ids are excluded.",
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    reference = {
        str(row["question_id"]): row
        for row in load_json_or_jsonl(args.reference)
    }
    candidate = {
        str(row["question_id"]): row
        for row in load_json_or_jsonl(args.candidate)
    }
    gt_rows = load_json_or_jsonl(args.gt)
    gt = {str(row["question_id"]): row["text"] for row in gt_rows}
    metadata = {}
    if args.question_meta:
        metadata = {
            str(row["question_id"]): row
            for row in load_json_or_jsonl(args.question_meta)
        }
    scene_by_question = {
        str(row["question_id"]): row.get(
            "scene_id",
            metadata.get(str(row["question_id"]), {}).get(
                "scene_id",
                metadata.get(str(row["question_id"]), {}).get(
                    "video",
                    str(row["question_id"]),
                ),
            ),
        )
        for row in gt_rows
    }

    excluded = set()
    if args.exclude:
        excluded = {
            str(row["question_id"])
            for row in load_json_or_jsonl(args.exclude)
            if "question_id" in row
        }
    question_ids = sorted((set(reference) & set(candidate) & set(gt)) - excluded)
    deltas = []
    scene_deltas = defaultdict(list)
    ref_correct = cand_correct = 0
    for question_id in question_ids:
        ref = exact_match(reference[question_id]["text"], gt[question_id])
        cand = exact_match(candidate[question_id]["text"], gt[question_id])
        delta = cand - ref
        ref_correct += ref
        cand_correct += cand
        deltas.append(delta)
        scene_deltas[scene_by_question[question_id]].append(delta)

    question_ci = bootstrap_question(deltas, args.iterations, args.seed)
    scene_ci = bootstrap_scene(scene_deltas, args.iterations, args.seed + 1)
    scale = 100.0
    print(f"n_questions={len(question_ids)} n_scenes={len(scene_deltas)}")
    print(f"reference_em={ref_correct / len(question_ids) * scale:.4f}")
    print(f"candidate_em={cand_correct / len(question_ids) * scale:.4f}")
    print(f"delta_pp={sum(deltas) / len(deltas) * scale:.4f}")
    print(f"question_bootstrap_95ci_pp=[{question_ci[0] * scale:.4f}, {question_ci[1] * scale:.4f}]")
    print(f"scene_bootstrap_95ci_pp=[{scene_ci[0] * scale:.4f}, {scene_ci[1] * scale:.4f}]")
    print(f"wins={sum(delta == 1 for delta in deltas)} "
          f"ties={sum(delta == 0 for delta in deltas)} "
          f"losses={sum(delta == -1 for delta in deltas)}")


if __name__ == "__main__":
    main()
