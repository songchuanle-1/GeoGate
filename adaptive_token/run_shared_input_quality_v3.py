# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# LLaVA-3D integration; see THIRD_PARTY.md for upstream attribution and terms.
"""Traced same-stack quality comparison, not an end-to-end timing benchmark."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import types

import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_videos, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from adaptive_token.geogate_v0 import enable_geogate, set_keep_ratio
from analysis.review_v3_protocol import rotation, sample_seed
from analysis.run_prompt_robustness_v3 import file_hash, seed_all, tensor_hash
from analysis.review_v3_model_provenance import provenance


class ForwardTrace:
    def __init__(self, model):
        self.calls = []
        self.layers = []
        self.model = model
        self.handles = [
            model.model.register_forward_hook(self.after_lm),
        ]
        for index, layer in enumerate(model.model.layers):
            def before(module, args, kwargs, index=index):
                hidden = args[0] if args else kwargs["hidden_states"]
                self.layers.append((index, int(hidden.shape[1])))
            self.handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))

    def after_lm(self, module, args, output):
        cache = output.past_key_values
        lengths = ([int(cache.get_seq_length(i)) for i in range(len(module.layers))]
                   if hasattr(cache, "get_seq_length")
                   else [int(layer[0].shape[-2]) for layer in cache])
        self.calls.append(lengths)

    def reset(self):
        self.calls.clear()
        self.layers.clear()

    def validate(self, full_length, expected_length, pruning):
        if not self.calls:
            raise ValueError("No language-model forward calls observed")
        expected = [full_length] * 15 + [expected_length] * 17 if pruning else [full_length] * 32
        if self.layers[:32] != list(enumerate(expected)):
            raise ValueError(f"Actual prefill layer lengths mismatch: {self.layers[:32]}")
        if self.calls[0] != [expected_length] * 32:
            raise ValueError(f"KV length mismatch: {self.calls[0]}")
        if any(length != 1 for _, length in self.layers[32:]):
            raise ValueError("Decode did not use single-token updates")
        for step, lengths in enumerate(self.calls):
            if lengths != [expected_length + step] * 32:
                raise ValueError("Decode KV lengths did not grow by exactly one")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
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
        expected_conditions = {"native", "zip128", "geogate128"}
        incomplete = {
            question_id: sorted(conditions)
            for question_id, conditions in conditions_by_question.items()
            if conditions != expected_conditions
        }
        if incomplete:
            raise ValueError(f"Existing output has incomplete triples: {incomplete}")
        completed = set(conditions_by_question)
    questions = json.loads(args.questions.read_text())
    if len(questions) != 4675 or len({r["question_id"] for r in questions}) != 4675:
        raise ValueError("Expected unique full ScanQA validation questions")
    if args.smoke:
        scenes = ("scene0011_00", "scene0231_00", "scene0249_00", "scene0342_00")
        questions = [r for scene in scenes
                     for r in [next(q for q in questions if q["video"].endswith(scene))]]
    seed_all(args.seed)
    native_tokenizer, native, processor, _ = load_pretrained_model(
        args.model, None, get_model_name_from_path(args.model), device_map="cuda:0")
    geo_tokenizer, geo, _, _ = load_pretrained_model(
        args.model, None, get_model_name_from_path(args.model), device_map="cuda:0")
    native.eval()
    geo.eval()
    if native.config.to_dict() != geo.config.to_dict():
        raise ValueError("Model configs differ")
    enable_geogate(geo, prune_layer=14, keep_ratio=0.1, scorer="attn")
    original_prepare = native.prepare_inputs_labels_for_multimodal
    geo_original_prepare = geo.prepare_inputs_labels_for_multimodal
    cached = {}

    def cached_prefill(pargs, kwargs):
        ids = pargs[0] if pargs else kwargs.get("input_ids")
        images = pargs[5] if len(pargs) > 5 else kwargs.get("images")
        return ids is not None and ids.shape[1] > 1 and images is not None

    def prepare_native(self, *pargs, **kwargs):
        if not cached_prefill(pargs, kwargs):
            return original_prepare(*pargs, **kwargs)
        return cached["current"]

    def prepare_geo(self, *pargs, **kwargs):
        if not cached_prefill(pargs, kwargs):
            return geo_original_prepare(*pargs, **kwargs)
        result = cached["native"]
        visual_count = cached["native_count"]
        kept = min(128, visual_count)
        geo.model._gg_span = (cached["sp0"], cached["sp0"] + visual_count)
        set_keep_ratio(geo, kept / visual_count)
        return result

    native.prepare_inputs_labels_for_multimodal = types.MethodType(prepare_native, native)
    geo.prepare_inputs_labels_for_multimodal = types.MethodType(prepare_geo, geo)
    traces = {"native": ForwardTrace(native), "geo": ForwardTrace(geo)}
    tower = native.get_model().get_video_tower()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "questions_sha256": file_hash(args.questions),
        "runner_sha256": file_hash(__file__),
        "core_sha256": file_hash(inspect.getfile(enable_geogate)),
        "core_path": inspect.getfile(enable_geogate),
        "llava_source_path": inspect.getfile(type(native)),
        "llava_source_sha256": file_hash(inspect.getfile(type(native))),
        "prepare_source_path": inspect.getfile(original_prepare),
        "prepare_source_sha256": file_hash(inspect.getfile(original_prepare)),
        "model_config_sha256": hashlib.sha256(native.config.to_json_string().encode()).hexdigest(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "seed": args.seed, "n_questions": len(questions),
        "conditions": ["native", "zip128", "geogate128"],
        "decoding": {"num_beams": 1, "do_sample": False, "max_new_tokens": 512},
        "timing": "not measured; precomputed inputs and trace instrumentation",
        "native_geogate_inputs": "same tensor object from native multimodal preparation",
        "provenance": provenance(native, args.model, original_prepare),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out.with_suffix(".manifest.json")
    if not args.resume:
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, indent=2)
    else:
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        previous = json.loads(manifest_path.read_text())
        for field in ("questions_sha256", "seed", "conditions"):
            if previous[field] != manifest[field]:
                raise ValueError(f"Resume manifest mismatch: {field}")
        resume_path = args.out.parent / (
            f"{args.out.stem}.resume_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        resume_path.write_text(json.dumps(
            manifest | {"resume_completed_questions": len(completed)}, indent=2))
    with args.out.open("a" if args.resume else "x", encoding="utf-8") as handle:
        for index, row in enumerate(questions):
            if str(row["question_id"]) in completed:
                continue
            seed = sample_seed(row["question_id"], args.seed)
            seed_all(seed)
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + row["text"])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            ids = tokenizer_image_token(prompt, native_tokenizer, IMAGE_TOKEN_INDEX,
                                        return_tensors="pt").unsqueeze(0).cuda()
            other_ids = tokenizer_image_token(prompt, geo_tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).cuda()
            if not torch.equal(ids, other_ids):
                raise ValueError("Tokenizers produce different prompt IDs")
            positions = (ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
            if positions.numel() != 1:
                raise ValueError("Expected one image placeholder")
            cached["sp0"] = int(positions[0])
            videos = process_videos(f"{args.video_folder}/{row['video']}", processor["video"],
                                    mode="uniform", device=native.device, text=row["text"])
            tensors = {key: videos[key].to(native.device, dtype=torch.bfloat16)
                       for key in ("images", "depths", "poses", "intrinsics")}
            input_hashes = {key: tensor_hash(value) for key, value in tensors.items()}
            count_before = {}
            with torch.inference_mode():
                for name, pooling, target in (("native", "voxelize", 0), ("zip128", "3dzip", 128)):
                    seed_all(seed)
                    tower.pooling, tower.adaptive_ratio, tower.voxel_size = pooling, target, 0.2
                    result = original_prepare(
                        ids, None, None, None, None,
                        tensors["images"], tensors["depths"], tensors["poses"],
                        tensors["intrinsics"], None, image_sizes=None)
                    cached[name] = result
                    count_before[name] = int(tower._voxel_xyz[0].shape[0])
            cached["native_count"] = int(cached["native"][4].shape[1] - ids.shape[1] + 1)
            native_hash = tensor_hash(cached["native"][4])
            question_records = []
            for order, condition in enumerate(rotation(("native", "zip128", "geogate128"), index)):
                pruning = condition == "geogate128"
                model, tokenizer, trace = ((geo, geo_tokenizer, traces["geo"]) if pruning else
                                           (native, native_tokenizer, traces["native"]))
                prepared = cached["native" if pruning else condition]
                cached["current"] = prepared
                length = int(prepared[4].shape[1])
                visual_count = length - ids.shape[1] + 1
                kept = min(128, visual_count) if pruning else visual_count
                expected_length = length - visual_count + kept
                trace.reset()
                seed_all(seed)
                with torch.inference_mode():
                    outputs = model.generate(
                        ids, **tensors, image_sizes=None, do_sample=False, num_beams=1,
                        max_new_tokens=512, use_cache=True)
                trace.validate(length, expected_length, pruning)
                selected = (geo.model._gg_orig_map.cpu().tolist() if pruning
                            else list(range(length)))
                if len(selected) != expected_length or selected != sorted(set(selected)):
                    raise ValueError("Selected original indices are invalid")
                if pruning:
                    sp0 = cached["sp0"]
                    actual_spatial = sum(sp0 <= i < sp0 + visual_count for i in selected)
                    if actual_spatial != kept:
                        raise ValueError("Actual spatial budget differs from target")
                embedding_hash = tensor_hash(prepared[4])
                if pruning and embedding_hash != native_hash:
                    raise ValueError("Shared embeddings changed")
                record = {
                    "question_id": row["question_id"], "scene_id": row["video"],
                    "condition": condition, "order": order, "seed": seed,
                    "prompt": row["text"], "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "text": tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip(),
                    "input_tensor_hashes": input_hashes,
                    "expanded_input_sha256": embedding_hash,
                    "visual_tokens_before_native_cap": count_before["native"],
                    "input_visual_tokens": visual_count, "actual_kept_spatial_tokens": kept,
                    "selected_original_indices": selected,
                    "prefill_layer_input_lengths": [length for _, length in trace.layers[:32]],
                    "prefill_kv_lengths": trace.calls[0],
                    "decode_forward_calls": len(trace.calls) - 1,
                    "single_token_decode_verified": True,
                }
                question_records.append(record)
            for record in question_records:
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            cached.clear()
            del prepared, result, outputs, tensors, videos
            torch.cuda.empty_cache()
            print(f"{index + 1}/{len(questions)} traced triples complete", flush=True)


if __name__ == "__main__":
    main()
