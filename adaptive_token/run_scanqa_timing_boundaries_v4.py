#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# LLaVA-3D integration; see THIRD_PARTY.md for upstream attribution and terms.
"""Full-set paired timing with explicit preprocessing, H2D, and decode boundaries."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_videos,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

from adaptive_token.geogate_v0 import enable_auto_span, enable_geogate


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_path):
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path,
        None,
        get_model_name_from_path(model_path),
        device_map="cuda:0",
    )
    model.eval()
    return tokenizer, model, processor


def compute_pids(gpu_id):
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_id),
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    return {int(line.strip()) for line in output.splitlines() if line.strip()}


def check_gpu(gpu_id, allowed_pids, context, require_idle=False):
    pids = compute_pids(gpu_id)
    unwanted = pids - set(allowed_pids)
    if unwanted:
        raise RuntimeError(f"{context}: foreign GPU PIDs on device {gpu_id}: {sorted(unwanted)}")
    if require_idle and pids:
        raise RuntimeError(f"{context}: GPU {gpu_id} is not idle: {sorted(pids)}")


def run_generate(model, tokenizer, input_ids, tensors):
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensors["images"],
            depths=tensors["depths"],
            poses=tensors["poses"],
            intrinsics=tensors["intrinsics"],
            image_sizes=None,
            do_sample=False,
            temperature=0,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )
    torch.cuda.synchronize()
    generation_s = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    decode_start = time.perf_counter()
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    decode_s = time.perf_counter() - decode_start
    mib = 1024 ** 2
    input_length = int(input_ids.shape[1])
    generated_length = int(output_ids.shape[1])
    return {
        "text": text,
        "generation_s": generation_s,
        "decode_s": decode_s,
        "input_tokens": input_length,
        "sequence_tokens": input_length + generated_length,
        "generated_tokens": generated_length,
        "allocated_before_mib": allocated_before / mib,
        "reserved_before_mib": reserved_before / mib,
        "peak_allocated_absolute_mib": peak_allocated / mib,
        "peak_allocated_incremental_mib": (peak_allocated - allocated_before) / mib,
        "peak_reserved_absolute_mib": peak_reserved / mib,
        "peak_reserved_incremental_mib": (peak_reserved - reserved_before) / mib,
    }


def exclusive_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--guard-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    for path in (args.out, args.manifest, args.status):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    check_gpu(args.gpu_id, set(), "preflight", require_idle=True)
    questions = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Question file must be a nonempty JSON array")
    ids = [str(row["question_id"]) for row in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs")
    if args.max_samples > 0:
        questions = questions[:args.max_samples]
    elif len(questions) != 4675:
        raise ValueError(f"Expected 4675 full ScanQA questions, found {len(questions)}")

    manifest = {
        "run_id": args.run_id,
        "session": args.session,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "gpu_id": args.gpu_id,
        "scope": (
            "paired single-H100 batch-1 timing on already inspected ScanQA validation; "
            "not an independent holdout"
        ),
        "model": {
            "path": str(Path(args.model_path).resolve()),
            "config_sha256": digest_file(Path(args.model_path) / "config.json"),
        },
        "questions": {
            "path": str(Path(args.question_file).resolve()),
            "sha256": digest_file(args.question_file),
            "count": len(questions),
        },
        "runner_sha256": digest_file(__file__),
        "protocol": {
            "batch_size": 1,
            "num_beams": 1,
            "do_sample": False,
            "max_new_tokens": 512,
            "warmup": args.warmup,
            "warmup_rows_emitted": False,
            "warmup_scope": (
                "first questions are repeated before measurement and excluded from JSONL"
            ),
            "seed": args.seed,
            "condition_order": "alternating by question index",
            "cuda_synchronize": True,
            "preprocessing_included_in_application": True,
            "generation_inputs": "resident GPU tensors",
            "geogate": {"prune_layer": 14, "keep_ratio": 0.10, "scorer": "attn"},
        },
    }
    exclusive_json(args.manifest, manifest)

    load_start = time.perf_counter()
    baseline_tokenizer, baseline, processor = load_model(args.model_path)
    baseline_load_s = time.perf_counter() - load_start
    load_start = time.perf_counter()
    geogate_tokenizer, geogate, _ = load_model(args.model_path)
    geogate_load_s = time.perf_counter() - load_start
    enable_geogate(geogate, prune_layer=14, keep_ratio=0.10, scorer="attn")
    enable_auto_span(geogate, IMAGE_TOKEN_INDEX)

    work_items = (
        [(row, True) for row in questions[:args.warmup]]
        + [(row, False) for row in questions]
    )
    with args.out.open("x", encoding="utf-8") as output:
        for index, (row, is_warmup) in enumerate(tqdm(work_items)):
            if index % args.guard_every == 0:
                check_gpu(args.gpu_id, {os.getpid()}, f"question {index}")
            measurement_index = index - args.warmup if not is_warmup else index
            question = row["text"]
            prompt_question = DEFAULT_IMAGE_TOKEN + "\n" + question
            conversation = conv_templates["llava_v1"].copy()
            conversation.append_message(conversation.roles[0], prompt_question)
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()
            input_ids = tokenizer_image_token(
                prompt,
                baseline_tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).cuda()

            seed_all(args.seed + measurement_index)
            preprocess_start = time.perf_counter()
            videos = process_videos(
                f"{args.video_folder}/{row['video']}",
                processor["video"],
                mode="random",
                device=baseline.device,
                text=question,
            )
            preprocess_s = time.perf_counter() - preprocess_start

            h2d_start = time.perf_counter()
            tensors = {
                key: videos[key].to(baseline.device, dtype=torch.bfloat16)
                for key in ("images", "depths", "poses", "intrinsics")
            }
            h2d_s = time.perf_counter() - h2d_start

            conditions = (
                (("baseline", baseline, baseline_tokenizer),
                 ("geogate", geogate, geogate_tokenizer))
                if measurement_index % 2 == 0
                else
                (("geogate", geogate, geogate_tokenizer),
                 ("baseline", baseline, baseline_tokenizer))
            )
            for order, (condition, model, tokenizer) in enumerate(conditions):
                result = run_generate(model, tokenizer, input_ids, tensors)
                application_s = (
                    preprocess_s + h2d_s + result["generation_s"] + result["decode_s"]
                )
                if not is_warmup:
                    output.write(json.dumps({
                        "run_id": args.run_id,
                        "session": args.session,
                        "question_id": row["question_id"],
                        "scene_id": row["video"],
                        "condition": condition,
                        "order": order,
                        "warmup": False,
                        "preprocess_s": preprocess_s,
                        "h2d_s": h2d_s,
                        "application_s": application_s,
                        **result,
                    }) + "\n")
                    output.flush()

    status = {
        "state": "finished",
        "run_id": args.run_id,
        "session": args.session,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(questions) * 2,
        "questions": len(questions),
        "warmup_questions": args.warmup,
        "baseline_load_s": baseline_load_s,
        "geogate_load_s": geogate_load_s,
        "raw_sha256": digest_file(args.out),
        "manifest_sha256": digest_file(args.manifest),
    }
    exclusive_json(args.status, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
