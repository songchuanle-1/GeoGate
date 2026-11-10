# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# LLaVA-3D integration; see THIRD_PARTY.md for upstream attribution and terms.
"""Frozen shared-embedding quality experiment; deliberately no timing claim."""

import argparse
import inspect
import importlib.metadata
import json
from pathlib import Path
import types

import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_videos, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from adaptive_token.geogate_v0 import enable_geogate, set_keep_ratio
from adaptive_token.run_shared_input_quality_v3 import ForwardTrace
from analysis.run_prompt_robustness_v3 import seed_all, tensor_hash
from analysis.review_v3_model_provenance import digest, provenance
from analysis.review_v3_protocol import rotation, sample_seed
from analysis.review_v4_journal import QuestionJournal


def validate_trace(trace, full, kept, layer):
    expected = ([full] * 32 if layer is None else
                [full] * (layer + 1) + [kept] * (31 - layer))
    if trace.layers[:32] != list(enumerate(expected)):
        raise ValueError("Incorrect prefill layer lengths")
    if not trace.calls or any(length != 1 for _, length in trace.layers[32:]):
        raise ValueError("Invalid decode input")
    for step, lengths in enumerate(trace.calls):
        if lengths != [kept + step] * 32:
            raise ValueError("Invalid per-layer KV growth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    questions = json.loads(args.questions.read_text())
    budget_doc = json.loads(args.budgets.read_text())
    budgets = {r["question_id"]: r for r in budget_doc["samples"]}
    ids = [str(r["question_id"]) for r in questions]
    if len(ids) != 4675 or len(set(ids)) != 4675 or set(ids) != set(budgets):
        raise ValueError("Question manifest mismatch")
    if args.smoke:
        scenes = ("scene0011_00", "scene0231_00", "scene0249_00", "scene0342_00")
        selected = {next(str(r["question_id"]) for r in questions
                         if r["video"].endswith(s)) for s in scenes}
    else:
        selected = set(ids)
    seed_all(20260908)
    tokenizer, native, processor, _ = load_pretrained_model(
        args.model, None, get_model_name_from_path(args.model), device_map="cuda:0")
    other_tokenizer, geo, _, _ = load_pretrained_model(
        args.model, None, get_model_name_from_path(args.model), device_map="cuda:0")
    native.eval()
    geo.eval()
    if native.config.to_dict() != geo.config.to_dict():
        raise ValueError("Model config mismatch")
    enable_geogate(geo, prune_layer=14, keep_ratio=0.1, scorer="attn")
    original = native.prepare_inputs_labels_for_multimodal
    geo_original = geo.prepare_inputs_labels_for_multimodal
    cached = {}

    def wrapper(fallback, pruning):
        def prepare(self, *pargs, **kwargs):
            tokens = pargs[0] if pargs else kwargs.get("input_ids")
            images = pargs[5] if len(pargs) > 5 else kwargs.get("images")
            if tokens is None or tokens.shape[1] <= 1 or images is None:
                return fallback(*pargs, **kwargs)
            if pruning:
                geo.model._gg_prune_layer = cached["layer"]
                set_keep_ratio(geo, cached["keep"] / cached["spatial"])
                geo.model._gg_span = (cached["start"], cached["start"] + cached["spatial"])
            return cached["prepared"]
        return prepare

    native.prepare_inputs_labels_for_multimodal = types.MethodType(wrapper(original, False), native)
    geo.prepare_inputs_labels_for_multimodal = types.MethodType(wrapper(geo_original, True), geo)
    traces = {False: ForwardTrace(native), True: ForwardTrace(geo)}
    conditions = ("native", "L2_cost", "L8_cost", "L14_128")
    manifest = {
        "seed": 20260908, "statistics_seed": 20260910,
        "questions_sha256": digest(args.questions), "budget_sha256": digest(args.budgets),
        "runner_sha256": digest(__file__),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "numpy")},
        "config": native.config.to_dict(),
        "helpers": {str(Path(inspect.getfile(obj)).resolve()): digest(inspect.getfile(obj))
                    for obj in (enable_geogate, ForwardTrace, QuestionJournal,
                                sample_seed, seed_all, process_videos)},
        "provenance": provenance(native, args.model, original),
        "decoding": {"max_new_tokens": 512, "num_beams": 1, "do_sample": False},
        "scope": "quality, analytic LM matched cost; not latency or holdout",
    }
    journal = QuestionJournal(args.out.with_suffix(".journal"), manifest,
                              [q for q in ids if q in selected], conditions)
    completed = journal.completed()
    tower = native.get_model().get_video_tower()
    tower.pooling, tower.adaptive_ratio, tower.voxel_size = "voxelize", 0, 0.2
    for index, row in enumerate(questions):
        qid = str(row["question_id"])
        if qid not in selected or qid in completed:
            continue
        seed = sample_seed(qid, 20260908)
        seed_all(seed)
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + row["text"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        tokens = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                       return_tensors="pt").unsqueeze(0).cuda()
        other = tokenizer_image_token(prompt, other_tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors="pt").unsqueeze(0).cuda()
        if not torch.equal(tokens, other):
            raise ValueError("Tokenizer mismatch")
        positions = (tokens[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
        if positions.numel() != 1:
            raise ValueError("Invalid spatial placeholder")
        start = int(positions[0])
        videos = process_videos(f"{args.video_folder}/{row['video']}", processor["video"],
                                mode="uniform", device=native.device, text=row["text"])
        tensors = {k: videos[k].to(native.device, dtype=torch.bfloat16)
                   for k in ("images", "depths", "poses", "intrinsics")}
        hashes = {k: tensor_hash(v) for k, v in tensors.items()}
        seed_all(seed)
        with torch.inference_mode():
            prepared = original(tokens, None, None, None, None, tensors["images"],
                                tensors["depths"], tensors["poses"], tensors["intrinsics"],
                                None, image_sizes=None)
        full = int(prepared[4].shape[1])
        spatial = full - tokens.shape[1] + 1
        budget = budgets[qid]
        if (spatial, full - spatial) != (budget["spatial"], budget["text"]):
            raise ValueError("Frozen cost lengths differ from actual inputs")
        embedding_hash = tensor_hash(prepared[4])
        cached.update(prepared=prepared, start=start, spatial=spatial)
        records = []
        for order, condition in enumerate(rotation(conditions, index)):
            layer = {"native": None, "L2_cost": 2, "L8_cost": 8, "L14_128": 14}[condition]
            pruning = layer is not None
            keep = (spatial if not pruning else 128 if layer == 14 else
                    budget["controls"][str(layer)]["keep"])
            cached.update(layer=layer, keep=keep)
            model, tok = (geo, other_tokenizer) if pruning else (native, tokenizer)
            trace = traces[pruning]
            trace.reset()
            seed_all(seed)
            with torch.inference_mode():
                generated = model.generate(tokens, **tensors, image_sizes=None,
                                           num_beams=1, do_sample=False,
                                           max_new_tokens=512, use_cache=True)
            length = full - spatial + keep
            validate_trace(trace, full, length, layer)
            indices = geo.model._gg_orig_map.cpu().tolist() if pruning else list(range(full))
            if (indices != sorted(set(indices)) or len(indices) != length
                    or sum(start <= i < start + spatial for i in indices) != keep
                    or any(i < 0 or i >= full for i in indices)):
                raise ValueError("Invalid actual selection")
            if tensor_hash(prepared[4]) != embedding_hash:
                raise ValueError("Shared input mutated")
            records.append({
                "question_id": qid, "scene_id": row["video"], "condition": condition,
                "order": order, "seed": seed, "prompt": prompt,
                "text": tok.batch_decode(generated, skip_special_tokens=True)[0].strip(),
                "input_tensor_hashes": hashes, "expanded_input_sha256": embedding_hash,
                "input_visual_tokens": spatial, "actual_kept_spatial_tokens": keep,
                "selected_original_indices": indices, "prune_layer": layer,
                "prefill_layer_input_lengths": [n for _, n in trace.layers[:32]],
                "prefill_kv_lengths": trace.calls[0],
                "decode_forward_calls": len(trace.calls) - 1,
                "single_token_decode_verified": True,
            })
        journal.commit(qid, records)
        cached.clear()
        del prepared, tensors, videos, generated
        torch.cuda.empty_cache()
        print(f"{index + 1}/4675 committed {qid}", flush=True)
    journal.export(args.out)


if __name__ == "__main__":
    main()
