# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# LLaVA-3D integration; see THIRD_PARTY.md for upstream attribution and terms.
"""Paired prompt profiles with shared visual embeddings and frozen sample IDs."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_videos, get_model_name_from_path
from llava.model.builder import load_pretrained_model

from analysis.review_v3_protocol import CONDITIONS, onset, prompt_variants, rotation, sample_seed
from analysis.review_v3_model_provenance import provenance


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gini(weights):
    weights = weights.double().sort().values
    if weights.sum() <= 0:
        return 0.0
    index = torch.arange(1, weights.numel() + 1, device=weights.device)
    return float(((2 * index - weights.numel() - 1) * weights).sum()
                 / (weights.numel() * weights.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.out.exists() and not args.resume:
        raise FileExistsError(args.out)
    completed = set()
    if args.resume and args.out.exists():
        conditions_by_question = {}
        with args.out.open(encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, 1):
                row = json.loads(line)
                question_id = str(row["question_id"])
                conditions = conditions_by_question.setdefault(question_id, set())
                if row["condition"] in conditions:
                    raise ValueError(f"Duplicate existing condition at line {line_number}")
                conditions.add(row["condition"])
        expected_conditions = set(CONDITIONS)
        incomplete = {
            question_id: sorted(conditions)
            for question_id, conditions in conditions_by_question.items()
            if conditions != expected_conditions
        }
        if incomplete:
            raise ValueError(f"Existing output has incomplete triples: {incomplete}")
        completed = set(conditions_by_question)
    source = json.loads(args.profile_manifest.read_text(encoding="utf-8"))
    samples = source["per_sample"]
    if len(samples) != 200 or len({s["video"] for s in samples}) != 67:
        raise ValueError("Expected the frozen 200-question / 67-scene profile")
    if len({s["question_id"] for s in samples}) != 200:
        raise ValueError("Duplicate manifest question")
    for row in samples:
        prompt_variants(row["text"])
    if args.max_samples:
        samples = samples[:args.max_samples]
    seed_all(args.seed)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model, None, get_model_name_from_path(args.model), device_map="cuda:0")
    model.eval()
    config_text = model.config.to_json_string()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args) | {"out": str(args.out), "profile_manifest": str(args.profile_manifest)},
        "profile_sha256": file_hash(args.profile_manifest),
        "runner_sha256": file_hash(__file__),
        "protocol_sha256": file_hash(inspect.getfile(prompt_variants)),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "model_path": args.model,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "numpy")},
        "source_sha256": {
            str(inspect.getfile(obj)): file_hash(inspect.getfile(obj))
            for obj in (type(model), type(model.model), process_videos)
        },
        "attention_backend": getattr(model.config, "_attn_implementation", None),
        "generation": False, "latency_scope": "instrumented profile, not generation speed",
        "conditions": list(CONDITIONS), "sample_count": len(samples),
        "provenance": provenance(model, args.model, model.prepare_inputs_labels_for_multimodal),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out.with_suffix(".manifest.json")
    config_path = args.out.parent / (args.out.stem + ".model_config.json")
    if not args.resume:
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        config_path.write_text(config_text)
    else:
        if not manifest_path.exists() or not config_path.exists():
            raise FileNotFoundError("Resume provenance files are missing")
        previous = json.loads(manifest_path.read_text())
        for field in ("profile_sha256", "config_sha256", "conditions"):
            if previous[field] != manifest[field]:
                raise ValueError(f"Resume manifest mismatch: {field}")
        resume_path = args.out.parent / (
            f"{args.out.stem}.resume_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        resume_path.write_text(json.dumps(
            manifest | {"resume_completed_questions": len(completed)}, indent=2))
    with args.out.open("a" if args.resume else "x", encoding="utf-8") as handle:
        for index, row in enumerate(samples):
            if str(row["question_id"]) in completed:
                continue
            per_seed = sample_seed(row["question_id"], args.seed)
            seed_all(per_seed)
            variants = prompt_variants(row["text"])

            def input_for(text):
                conv = conv_templates["llava_v1"].copy()
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                            return_tensors="pt").unsqueeze(0).cuda()
                return ids, prompt

            original_ids, _ = input_for(variants["original"])
            videos = process_videos(f"{args.video_folder}/{row['video']}", processor["video"],
                                    mode="random", device=model.device, text=row["text"])
            tensors = {key: videos[key].to(model.device, dtype=torch.bfloat16)
                       for key in ("images", "depths", "poses", "intrinsics")}
            input_hashes = {key: tensor_hash(value) for key, value in tensors.items()}
            positions = (original_ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
            if positions.numel() != 1:
                raise ValueError("Exactly one image placeholder required")
            sp0 = int(positions[0])
            # Encode the scene once. All three conditions reuse identical spatial embeddings.
            with torch.inference_mode():
                prepared = model.prepare_inputs_labels_for_multimodal(
                    original_ids, None, None, None, None, tensors["images"],
                    tensors["depths"], tensors["poses"], tensors["intrinsics"],
                    None, image_sizes=None)
                expanded = prepared[4]
                n_spatial = int(expanded.shape[1] - original_ids.shape[1] + 1)
                visual = expanded[:, sp0:sp0 + n_spatial, :].detach().clone()
                reconstructed = torch.cat((
                    model.model.embed_tokens(original_ids[:, :sp0]), visual,
                    model.model.embed_tokens(original_ids[:, sp0 + 1:]),
                ), dim=1)
                if not torch.equal(expanded, reconstructed):
                    raise ValueError("Shared-input reconstruction differs from native preparation")
            if n_spatial <= 0:
                raise ValueError("Empty spatial span")
            visual_hash = tensor_hash(visual)
            del expanded, reconstructed, prepared, tensors, videos
            question_records = []
            for order, condition in enumerate(rotation(CONDITIONS, index)):
                ids, prompt = input_for(variants[condition])
                position = (ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
                if position.numel() != 1 or int(position[0]) != sp0:
                    raise ValueError("Prompt rewrite changed spatial prefix")
                with torch.inference_mode():
                    prefix = model.model.embed_tokens(ids[:, :sp0])
                    suffix = model.model.embed_tokens(ids[:, sp0 + 1:])
                    embeds = torch.cat((prefix, visual, suffix), dim=1)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = model.model(inputs_embeds=embeds, output_attentions=True,
                                         use_cache=False, return_dict=True)
                    torch.cuda.synchronize()
                    latency = time.perf_counter() - start
                record = {
                    "question_id": row["question_id"], "scene_id": row["video"],
                    "condition": condition, "order": order, "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "seed": per_seed, "input_tensor_hashes": input_hashes,
                    "visual_embedding_sha256": visual_hash,
                    "n_spatial": n_spatial, "span_start": sp0,
                    "total_tokens": int(embeds.shape[1]), "profile_forward_s": latency,
                    "frac": [], "top50": [], "last_prompt_gini": [], "topk_indices": [],
                }
                for attention in output.attentions:
                    if attention is None:
                        raise ValueError("Attention backend did not return weights")
                    weights = attention[0, :, -1, :].float().mean(dim=0)
                    spatial = weights[sp0:sp0 + n_spatial]
                    sorted_weights, indices = torch.sort(spatial, descending=True, stable=True)
                    if spatial.sum() <= 0:
                        raise ValueError("Zero spatial attention mass")
                    coverage = int((sorted_weights.cumsum(0) < spatial.sum() * 0.5).sum()) + 1
                    k = min(n_spatial, max(1, round(n_spatial * 0.10)))
                    record["frac"].append(float(spatial.sum() / weights.sum()))
                    record["top50"].append(coverage / n_spatial)
                    record["last_prompt_gini"].append(gini(spatial))
                    record["topk_indices"].append(indices[:k].cpu().tolist())
                if len(record["frac"]) != 32:
                    raise ValueError("Protocol expects 32 layers")
                record["onset"] = onset(record["frac"], record["top50"])
                record["l14_l13_jump"] = record["frac"][14] - record["frac"][13]
                question_records.append(record)
                del output, embeds, prefix, suffix, attention, weights, spatial, sorted_weights
            for record in question_records:
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            del visual
            torch.cuda.empty_cache()
            print(f"{index + 1}/{len(samples)} paired profiles complete", flush=True)


if __name__ == "__main__":
    main()
