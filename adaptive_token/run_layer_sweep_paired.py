# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# LLaVA-3D integration; see THIRD_PARTY.md for upstream attribution and terms.
"""Shared-input paired ScanQA sweep over the frozen layer curve candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import time
import types

import numpy as np
import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_videos,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

from adaptive_token.geogate_v0 import enable_geogate, set_keep_ratio
from adaptive_token.gpu_guard import guard_gpu
from adaptive_token.forward_trace import ForwardTrace
from analysis.layer_sweep_protocol import DEFAULT_LAYERS, parse_layers
from analysis.review_v3_model_provenance import digest, provenance
from analysis.review_v3_protocol import rotation, sample_seed
from analysis.review_v4_journal import QuestionJournal


SMOKE_SCENES = (
    "scene0011_00",
    "scene0231_00",
    "scene0249_00",
    "scene0342_00",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def validate_trace(
    trace: ForwardTrace,
    *,
    full: int,
    kept: int,
    layer: int | None,
) -> None:
    expected = (
        [full] * 32
        if layer is None
        else [full] * (layer + 1) + [kept] * (31 - layer)
    )
    if trace.layers[:32] != list(enumerate(expected)):
        raise ValueError(
            f"Layer-trace mismatch: {trace.layers[:32]} != {list(enumerate(expected))}"
        )
    if not trace.calls or any(length != 1 for _, length in trace.layers[32:]):
        raise ValueError("Decode did not use single-token updates")
    for step, lengths in enumerate(trace.calls):
        if lengths != [kept + step] * 32:
            raise ValueError("Decode KV lengths did not grow by exactly one")


def select_questions(rows: list[dict], smoke: bool) -> list[dict]:
    if smoke:
        selected = []
        for scene in SMOKE_SCENES:
            matches = [row for row in rows if str(row["video"]).endswith(scene)]
            if not matches:
                raise ValueError(f"Smoke scene is absent: {scene}")
            selected.append(matches[0])
        return selected
    if len(rows) != 4675 or len({str(row["question_id"]) for row in rows}) != 4675:
        raise ValueError("Expected 4,675 unique ScanQA validation questions")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--statistics-seed", type=int, default=20260920)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--guard-every", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.guard_every < 1:
        raise ValueError("--guard-every must be positive")
    if not 0.0 < args.keep_ratio <= 1.0:
        raise ValueError("--keep-ratio must lie in (0, 1]")
    layers = parse_layers(args.layers)
    conditions = ("native",) + tuple(f"L{layer}" for layer in layers)
    if args.gpu_id is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if len(devices) != 1:
            raise ValueError(
                "--gpu-id is required unless exactly one CUDA device is visible"
            )
        args.gpu_id = int(devices[0])

    rows = json.loads(args.questions.read_text(encoding="utf-8"))
    selected_rows = select_questions(rows, args.smoke)
    selected_ids = [str(row["question_id"]) for row in selected_rows]
    guard_gpu(args.gpu_id, {os.getpid()}, "layer sweep before model load")

    tokenizer, native, processor, _ = load_pretrained_model(
        args.model,
        None,
        get_model_name_from_path(args.model),
        device_map="cuda:0",
    )
    other_tokenizer, geo, _, _ = load_pretrained_model(
        args.model,
        None,
        get_model_name_from_path(args.model),
        device_map="cuda:0",
    )
    native.eval()
    geo.eval()
    if native.config.to_dict() != geo.config.to_dict():
        raise ValueError("Model config mismatch")
    enable_geogate(
        geo,
        prune_layer=layers[0],
        keep_ratio=args.keep_ratio,
        scorer="attn",
    )
    original = native.prepare_inputs_labels_for_multimodal
    geo_original = geo.prepare_inputs_labels_for_multimodal
    cached: dict = {}

    def wrapper(fallback, pruning: bool):
        def prepare(self, *pargs, **kwargs):
            tokens = pargs[0] if pargs else kwargs.get("input_ids")
            images = pargs[5] if len(pargs) > 5 else kwargs.get("images")
            if tokens is None or tokens.shape[1] <= 1 or images is None:
                return fallback(*pargs, **kwargs)
            if pruning:
                geo.model._gg_prune_layer = cached["layer"]
                set_keep_ratio(geo, cached["keep"] / cached["spatial"])
                geo.model._gg_span = (
                    cached["start"],
                    cached["start"] + cached["spatial"],
                )
            return cached["prepared"]

        return prepare

    native.prepare_inputs_labels_for_multimodal = types.MethodType(
        wrapper(original, False),
        native,
    )
    geo.prepare_inputs_labels_for_multimodal = types.MethodType(
        wrapper(geo_original, True),
        geo,
    )
    traces = {False: ForwardTrace(native), True: ForwardTrace(geo)}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema": "geogate-layer-sweep-paired-v1",
        "seed": args.seed,
        "statistics_seed": args.statistics_seed,
        "keep_ratio": args.keep_ratio,
        "layers": list(layers),
        "conditions": list(conditions),
        "max_new_tokens": args.max_new_tokens,
        "questions_sha256": digest(args.questions),
        "runner_sha256": digest(__file__),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy")
        },
        "config": native.config.to_dict(),
        "helpers": {
            str(Path(inspect.getfile(obj)).resolve()): digest(inspect.getfile(obj))
            for obj in (
                enable_geogate,
                ForwardTrace,
                QuestionJournal,
                sample_seed,
                process_videos,
            )
        },
        "provenance": provenance(native, args.model, original),
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": 1,
            "do_sample": False,
        },
        "scope": (
            "Paired shared-input rerun of the measured 11-layer curve. "
            "It is not an independent holdout and does not replace the "
            "paper's earlier reported values."
        ),
    }
    journal = QuestionJournal(
        args.out.with_suffix(".journal"),
        manifest,
        selected_ids,
        conditions,
    )
    completed = journal.completed()
    tower = native.get_model().get_video_tower()
    tower.pooling, tower.adaptive_ratio, tower.voxel_size = "voxelize", 0, 0.2

    for index, row in enumerate(selected_rows):
        question_id = str(row["question_id"])
        if question_id in completed:
            continue
        if index == 0 or (index + 1) % args.guard_every == 0:
            guard_gpu(args.gpu_id, {os.getpid()}, f"layer sweep question {index + 1}")
        seed = sample_seed(question_id, args.seed)
        seed_all(seed)
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(
            conv.roles[0],
            DEFAULT_IMAGE_TOKEN + "\n" + row["text"],
        )
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        tokens = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).cuda()
        other = tokenizer_image_token(
            prompt,
            other_tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).cuda()
        if not torch.equal(tokens, other):
            raise ValueError("Tokenizer mismatch")
        positions = (tokens[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
        if positions.numel() != 1:
            raise ValueError("Exactly one spatial placeholder is required")
        start = int(positions[0])
        videos = process_videos(
            f"{args.video_folder}/{row['video']}",
            processor["video"],
            mode="random",
            device=native.device,
            text=row["text"],
        )
        tensors = {
            key: videos[key].to(native.device, dtype=torch.bfloat16)
            for key in ("images", "depths", "poses", "intrinsics")
        }
        hashes = {key: tensor_hash(value) for key, value in tensors.items()}
        seed_all(seed)
        with torch.inference_mode():
            prepared = original(
                tokens,
                None,
                None,
                None,
                None,
                tensors["images"],
                tensors["depths"],
                tensors["poses"],
                tensors["intrinsics"],
                None,
                image_sizes=None,
            )
        full = int(prepared[4].shape[1])
        spatial = full - tokens.shape[1] + 1
        if spatial < 1:
            raise ValueError("Empty spatial span")
        embedding_hash = tensor_hash(prepared[4])
        cached.update(prepared=prepared, start=start, spatial=spatial)
        records = []
        for order, condition in enumerate(rotation(conditions, index)):
            pruning = condition != "native"
            layer = None if not pruning else int(condition[1:])
            keep = (
                spatial
                if not pruning
                else max(1, int(round(args.keep_ratio * spatial)))
            )
            cached.update(layer=layer, keep=keep)
            model, tok = (geo, other_tokenizer) if pruning else (native, tokenizer)
            trace = traces[pruning]
            trace.reset()
            seed_all(seed)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    tokens,
                    **tensors,
                    image_sizes=None,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            length = full - spatial + keep
            validate_trace(trace, full=full, kept=length, layer=layer)
            indices = (
                list(range(full))
                if not pruning
                else geo.model._gg_orig_map.cpu().tolist()
            )
            if (
                indices != sorted(set(indices))
                or len(indices) != length
                or any(index < 0 or index >= full for index in indices)
                or sum(start <= index < start + spatial for index in indices) != keep
            ):
                raise ValueError("Invalid selected original indices")
            if tensor_hash(prepared[4]) != embedding_hash:
                raise ValueError("Shared input mutated")
            records.append(
                {
                    "question_id": question_id,
                    "scene_id": str(row["video"]),
                    "condition": condition,
                    "order": order,
                    "seed": seed,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "text": tok.batch_decode(
                        generated,
                        skip_special_tokens=True,
                    )[0].strip(),
                    "latency_s": elapsed,
                    "input_tensor_hashes": hashes,
                    "expanded_input_sha256": embedding_hash,
                    "input_visual_tokens": spatial,
                    "span_start": start,
                    "text_tokens": full - spatial,
                    "full_tokens": full,
                    "actual_kept_spatial_tokens": keep,
                    "selected_original_indices": indices,
                    "prune_layer": layer,
                    "keep_ratio": (
                        1.0
                        if not pruning
                        else float(keep) / float(spatial)
                    ),
                    "prefill_layer_input_lengths": [
                        n for _, n in trace.layers[:32]
                    ],
                    "prefill_kv_lengths": trace.calls[0],
                    "decode_forward_calls": len(trace.calls) - 1,
                    "single_token_decode_verified": True,
                }
            )
        journal.commit(question_id, records)
        cached.clear()
        del prepared, tensors, videos, generated
        torch.cuda.empty_cache()
        print(
            f"{index + 1}/{len(selected_rows)} committed {question_id}",
            flush=True,
        )
    journal.export(args.out)


if __name__ == "__main__":
    main()
