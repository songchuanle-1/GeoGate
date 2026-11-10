# SPDX-License-Identifier: Apache-2.0
"""Deterministic scene-level replacement for LLaVA-3D's random visual cap."""

import hashlib
import types

import torch


DEFAULT_CAP_SIZE = 2560


def scene_cap_seed(scene_id, scene_seed, token_count):
    payload = (
        f"3dvlm-scene-cap-v1\0{scene_id}\0{int(scene_seed)}\0"
        f"{int(token_count)}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    ) % (2 ** 63 - 1)


def tensor_sha256(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def deterministic_permutation(
    token_count,
    scene_id,
    scene_seed,
    randperm=torch.randperm,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        scene_cap_seed(scene_id, scene_seed, token_count)
    )
    return randperm(int(token_count), generator=generator)


def set_deterministic_scene_cap(model, scene_id, scene_seed):
    if not getattr(model, "_deterministic_scene_cap_enabled", False):
        raise RuntimeError("Deterministic scene cap is not enabled")
    model._deterministic_scene_cap_context = {
        "scene_id": str(scene_id),
        "scene_seed": int(scene_seed),
    }


def enable_deterministic_scene_cap(model, cap_size=DEFAULT_CAP_SIZE):
    """Share a reproducible visual-token cap across slots and model calls."""
    if getattr(model, "_deterministic_scene_cap_enabled", False):
        if model._deterministic_scene_cap_size != int(cap_size):
            raise ValueError("Deterministic scene cap size cannot be changed")
        return model

    original_prepare = model.prepare_inputs_labels_for_multimodal
    model._deterministic_scene_cap_size = int(cap_size)
    model._deterministic_scene_cap_context = None
    model._deterministic_scene_cap_trace = []

    def prepare_with_scene_cap(self, *args, **kwargs):
        context = self._deterministic_scene_cap_context
        if context is None:
            raise RuntimeError(
                "Set the deterministic scene-cap context before generation"
            )
        input_ids = args[0] if args else kwargs.get("input_ids")
        is_prefill = (
            input_ids is None
            or not torch.is_tensor(input_ids)
            or input_ids.shape[1] > 1
        )

        original_randperm = torch.randperm
        cache = {}
        trace = []

        def fixed_randperm(token_count, *rand_args, **rand_kwargs):
            token_count = int(token_count)
            if token_count <= self._deterministic_scene_cap_size:
                return original_randperm(
                    token_count,
                    *rand_args,
                    **rand_kwargs,
                )
            if rand_args or rand_kwargs:
                raise RuntimeError(
                    "Unexpected randperm arguments in the visual-cap path"
                )
            key = (
                context["scene_id"],
                context["scene_seed"],
                token_count,
            )
            if key not in cache:
                cache[key] = deterministic_permutation(
                    token_count,
                    context["scene_id"],
                    context["scene_seed"],
                    randperm=original_randperm,
                )
            permutation = cache[key].clone()
            trace.append({
                "scene_id": context["scene_id"],
                "scene_seed": context["scene_seed"],
                "token_count_before_cap": token_count,
                "cap_size": self._deterministic_scene_cap_size,
                "permutation_sha256": tensor_sha256(permutation),
                "selected_sha256": tensor_sha256(
                    permutation[:self._deterministic_scene_cap_size]
                ),
            })
            return permutation

        torch.randperm = fixed_randperm
        try:
            result = original_prepare(*args, **kwargs)
        finally:
            torch.randperm = original_randperm
        if is_prefill:
            self._deterministic_scene_cap_trace = trace
        return result

    model.prepare_inputs_labels_for_multimodal = types.MethodType(
        prepare_with_scene_cap,
        model,
    )
    model._deterministic_scene_cap_enabled = True
    return model


def audit_scene_cap_trace(model, expected_slots, expected_token_count):
    trace = list(model._deterministic_scene_cap_trace)
    if int(expected_token_count) <= model._deterministic_scene_cap_size:
        if trace:
            raise RuntimeError(
                "Visual-cap trace exists for an uncapped scene"
            )
        return {
            "applied": False,
            "slots": int(expected_slots),
            "token_count_before_cap": int(expected_token_count),
            "cap_size": model._deterministic_scene_cap_size,
            "selected_sha256": None,
        }
    if len(trace) != int(expected_slots):
        raise RuntimeError(
            f"Expected {expected_slots} visual-cap calls, found {len(trace)}"
        )
    token_counts = {row["token_count_before_cap"] for row in trace}
    selected = {row["selected_sha256"] for row in trace}
    if token_counts != {int(expected_token_count)}:
        raise RuntimeError(
            f"Unexpected pre-cap token counts: {sorted(token_counts)}"
        )
    if len(selected) != 1:
        raise RuntimeError("Batch slots used different visual-cap indices")
    first = trace[0]
    return {
        "applied": True,
        "slots": len(trace),
        "token_count_before_cap": first["token_count_before_cap"],
        "cap_size": first["cap_size"],
        "selected_sha256": first["selected_sha256"],
    }
