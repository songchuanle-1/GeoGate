# SPDX-License-Identifier: Apache-2.0
# Llama forward/cache interface adapted from Hugging Face Transformers (Apache-2.0).
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# GeoGate modifications include fixed-set spatial deletion and prompt-cache compaction.
"""GeoGate v0 variant with fixed-token-set deletion support (v21).

v0 scope (single-drop, FastV-comparable):
  * During prefill, at layer `prune_layer`, score spatial tokens by the
    attention the last prompt token pays to them (head-averaged, eager attn),
    keep top `keep_ratio` fraction, drop the rest for all subsequent layers.
  * After prefill, KV caches of ALL layers are pruned to the same kept set so
    decode-time masks stay consistent (standard FastV-style approximation).
  * Non-spatial (system/prompt) tokens are always kept.

E1-informed defaults: prune_layer=14 (readout onset), keep_ratio=0.10.
Set spatial span per sample via `set_span(model, sp0, sp1)` before generate.

Usage:
    from adaptive_token.geogate_v0 import enable_geogate, set_span
    enable_geogate(model, prune_layer=14, keep_ratio=0.10)
    set_span(model, sp0, sp1)   # per sample
    model.generate(...)

Adaptive mode:
    enable_geogate(model, prune_layer=-1, adaptive_onset=True,
                   scorer='stable_attn')
    set_span(model, sp0, sp1)
"""
import types

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast


def snapshot_indices(snapshot, final_map):
    """Map sorted original-token ids in final_map into a snapshot."""
    idx = torch.searchsorted(snapshot, final_map)
    if idx.numel() and (
        torch.any(idx >= snapshot.numel())
        or not torch.equal(snapshot[idx], final_map)
    ):
        raise ValueError("final token map is not contained in KV snapshot")
    return idx


def reset_adaptive_state(lm):
    """Reset per-sample detector state at the start of each prefill."""
    lm._gg_onset_layer = None
    lm._gg_onset_hits = 0
    lm._gg_attn_history = []
    lm._gg_stages = {-1: lm._gg_keep_ratio}


def enable_auto_span(model, image_token_index):
    """Infer the expanded spatial span during the real multimodal prefill."""
    if getattr(model, '_gg_auto_span_enabled', False):
        return model
    original_prepare = model.prepare_inputs_labels_for_multimodal
    lm = model.model

    def prepare_with_span(self, *args, **kwargs):
        result = original_prepare(*args, **kwargs)
        input_ids = args[0] if args else kwargs.get('input_ids')
        inputs_embeds = result[4]
        if (
            torch.is_tensor(input_ids)
            and torch.is_tensor(inputs_embeds)
            and input_ids.ndim == 2
        ):
            image_positions = (input_ids[0] == image_token_index).nonzero(
                as_tuple=False
            )
            if image_positions.numel():
                image_pos = int(image_positions[0, 0].item())
                n_spatial = inputs_embeds.shape[1] - (input_ids.shape[1] - 1)
                if n_spatial > 0:
                    lm._gg_span = (image_pos, image_pos + int(n_spatial))
        return result

    model.prepare_inputs_labels_for_multimodal = types.MethodType(
        prepare_with_span, model
    )
    model._gg_auto_span_enabled = True
    return model


def set_span(model, sp0, sp1):
    model.model._gg_span = (sp0, sp1)


def set_keep_ratio(model, keep_ratio):
    """Update the per-sample budget without re-patching the model."""
    lm = model.model
    lm._gg_keep_ratio = float(keep_ratio)
    if not getattr(lm, '_gg_stages_user_set', False):
        lm._gg_stages = {lm._gg_prune_layer: float(keep_ratio)}


def set_fixed_spatial_deletion(model, indices):
    """Inject a precomputed spatial deletion set for the next prefill.

    `indices` are absolute positions in the *current* full-length sequence
    (i.e. original indices before any pruning), matching the span
    `[sp0, sp1)`. When set, `_gg_forward` deletes exactly this set instead
    of scoring the current layer's attention. The set is consumed by the
    next prefill and reset to `None`.
    """
    lm = model.model
    if indices is None:
        lm._gg_fixed_spatial_deletion = None
    else:
        lm._gg_fixed_spatial_deletion = torch.as_tensor(
            indices,
            dtype=torch.long,
        )
    return model


def enable_geogate(model, prune_layer=14, keep_ratio=0.10, coarse_groups=0,
                   geo_beta=0.0, geo_grouping=False, stages=None, scorer='attn',
                   scoring=None, adaptive_onset=False, stability_window=3,
                   min_onset_layer=10, max_onset_layer=18, onset_patience=2,
                   onset_min_alpha=0.10, onset_max_coverage=0.12,
                   coverage_weight=0.25, prune_mode='keep_topk', delete_n=0):
    """Patch the LlamaModel instance inside a LlavaLlamaForCausalLM.

    coarse_groups=0  -> v0 behaviour: dropped spatial tokens are discarded.
    coarse_groups=G  -> HiToken: dropped tokens are mean-pooled into G coarse
                        tokens (contiguous groups, median position), and every
                        layer's KV cache is pooled the same way, so background
                        context survives at 1/G-th the cost.
    """
    lm = model.model  # LlamaModel subclass
    lm._gg_prune_layer = prune_layer
    lm._gg_keep_ratio = keep_ratio
    lm._gg_coarse_groups = coarse_groups
    lm._gg_geo_beta = geo_beta
    lm._gg_geo_grouping = geo_grouping
    lm._gg_scorer = scorer
    lm._gg_adaptive_onset = bool(adaptive_onset)
    lm._gg_stability_window = max(1, int(stability_window))
    lm._gg_min_onset_layer = max(0, int(min_onset_layer))
    lm._gg_max_onset_layer = min(len(lm.layers) - 1, int(max_onset_layer))
    lm._gg_onset_patience = max(1, int(onset_patience))
    lm._gg_onset_min_alpha = float(onset_min_alpha)
    lm._gg_onset_max_coverage = float(onset_max_coverage)
    lm._gg_coverage_weight = float(coverage_weight)
    if prune_mode not in ('keep_topk', 'delete_topk'):
        raise ValueError(f'unsupported prune_mode: {prune_mode}')
    lm._gg_prune_mode = prune_mode
    lm._gg_delete_n = max(0, int(delete_n))
    # scoring: optional callable(attn, sp0, sp1, hidden_states=...) -> scores
    # (compat with scoring_baselines.SCORING_METHODS from prior session)
    lm._gg_scoring = scoring
    # stages: {layer: keep_ratio_of_current_spatial}; overrides single-drop
    lm._gg_stages = dict(stages) if stages else {prune_layer: keep_ratio}
    lm._gg_stages_user_set = bool(stages)
    lm._gg_coords = None
    lm._gg_fixed_spatial_deletion = None
    lm._gg_span = None
    lm._gg_maxpos = 0
    lm._gg_next_pos = None
    lm._gg_onset_layer = None
    lm._gg_attn_history = []
    lm._gg_onset_hits = 0
    # keep ORIGINAL RoPE positions after pruning: rotary cache must cover the
    # original sequence length even though kv_seq_len shrank -> wrap rotary_emb
    for layer in lm.layers:
        rot = layer.self_attn.rotary_emb
        if getattr(rot, '_gg_wrapped', False):
            continue
        orig_fwd = rot.forward

        def wrapped(x, seq_len=None, _orig=orig_fwd, _lm=lm):
            need = max(seq_len or 0, _lm._gg_maxpos)
            cos, sin = _orig(x, seq_len=need)
            return cos, sin

        rot.forward = wrapped
        rot._gg_wrapped = True
    lm.forward = types.MethodType(_gg_forward, lm)
    # after pruning, HF slices decode inputs by (mask_len - cache_len) which
    # snowballs into re-feeding ALL generated tokens each step (and RoPE OOB);
    # force one-token decode steps for pruned samples
    if not getattr(model, '_gg_pifg_wrapped', False):
        orig_pifg = model.prepare_inputs_for_generation

        def pifg(input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
            out = orig_pifg(input_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, **kwargs)
            if past_key_values is not None and lm._gg_next_pos is not None:
                ii = out.get('input_ids')
                if ii is not None and ii.shape[1] > 1:
                    out['input_ids'] = ii[:, -1:]
                if out.get('position_ids') is not None:
                    out['position_ids'] = None  # our forward rebuilds them
            return out

        model.prepare_inputs_for_generation = pifg
        model._gg_pifg_wrapped = True
    return model


def _gg_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
    # mirrors transformers 4.37.2 LlamaModel.forward, plus single-drop pruning
    from transformers.cache_utils import Cache, DynamicCache

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states if output_hidden_states is not None
                            else self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError('specify either input_ids or inputs_embeds')
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    batch, seq_len, _ = inputs_embeds.shape
    assert batch == 1, 'GeoGate v0 supports batch=1 inference'

    past_len = 0
    use_legacy_cache = False
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_len = past_key_values.get_usable_length(seq_len)

    if position_ids is None:
        position_ids = torch.arange(past_len, seq_len + past_len,
                                    dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

    # prefill = no cache yet (decode steps may feed >1 token when HF catches up
    # after cache pruning, so seq_len>1 is NOT a reliable prefill signal)
    is_prefill = past_len == 0
    prune_here = (is_prefill and self._gg_span is not None
                  and (self._gg_adaptive_onset or
                       any(0 <= k < len(self.layers) for k in self._gg_stages)))
    if is_prefill:
        if self._gg_adaptive_onset:
            reset_adaptive_state(self)
        self._gg_maxpos = seq_len + 640
        self._gg_next_pos = None
        dev0 = inputs_embeds.device
        self._gg_orig_map = torch.arange(seq_len, device=dev0)
        self._gg_snapshots = {seq_len: self._gg_orig_map.clone()}

    # decode steps after pruning: generate() keeps growing the ORIGINAL-length
    # 2D mask; always rebuild from the REAL cache length
    if not is_prefill and use_cache:
        real_past = past_key_values.get_seq_length()
        past_len = real_past
        attention_mask = torch.ones((batch, real_past + seq_len),
                                    dtype=torch.long, device=inputs_embeds.device)
        # positions continue from the ORIGINAL sequence, not the pruned length
        if self._gg_next_pos is not None:
            position_ids = torch.arange(self._gg_next_pos, self._gg_next_pos + seq_len,
                                        device=inputs_embeds.device).unsqueeze(0)
            self._gg_next_pos += seq_len
        else:
            position_ids = torch.arange(real_past, real_past + seq_len,
                                        device=inputs_embeds.device).unsqueeze(0)

    # always use eager-style explicit 4D mask so per-layer attentions are valid
    attn_mask_4d = _prepare_4d_causal_attention_mask(
        attention_mask, (batch, seq_len), inputs_embeds, past_len)

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_attns = () if output_attentions else None
    kept_idx_full = None

    for li, layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        need_adaptive_attn = (
            prune_here and self._gg_adaptive_onset
            and self._gg_onset_layer is None
            and self._gg_min_onset_layer <= li <= self._gg_max_onset_layer
        )
        need_stable_attn = (
            prune_here and not self._gg_adaptive_onset
            and self._gg_scorer == 'stable_attn'
            and len(self._gg_stages) == 1
            and self._gg_prune_layer >= 0
            and max(0, self._gg_prune_layer - self._gg_stability_window + 1)
            <= li <= self._gg_prune_layer
        )
        has_fixed_deletion = (
            prune_here
            and getattr(self, '_gg_fixed_spatial_deletion', None)
            is not None
            and li in self._gg_stages
        )
        need_attn = (
            output_attentions or need_adaptive_attn or need_stable_attn
            or (prune_here and li in self._gg_stages
                and not has_fixed_deletion)
        )
        layer_out = layer(hidden_states, attention_mask=attn_mask_4d,
                          position_ids=position_ids, past_key_value=past_key_values,
                          output_attentions=need_attn, use_cache=use_cache)
        hidden_states = layer_out[0]
        if output_attentions:
            all_attns += (layer_out[1],)

        if need_stable_attn and layer_out[1] is not None:
            sp0, sp1 = self._gg_span
            self._gg_attn_history.append(
                layer_out[1][0].float().mean(dim=0)[-1, sp0:sp1].detach()
            )
            self._gg_attn_history = self._gg_attn_history[-self._gg_stability_window:]

        should_prune = (
            prune_here
            and not self._gg_adaptive_onset
            and li in self._gg_stages
        )
        if (prune_here and self._gg_adaptive_onset and need_adaptive_attn
                and self._gg_onset_layer is None):
            sp0, sp1 = self._gg_span
            attn_for_policy = layer_out[1][0].float().mean(dim=0)
            spatial_attn = attn_for_policy[-1, sp0:sp1]
            spatial_mass = spatial_attn.sum().clamp_min(torch.finfo(spatial_attn.dtype).eps)
            alpha = float(spatial_mass / attn_for_policy[-1].sum().clamp_min(1e-8))
            top_k = min(50, spatial_attn.numel())
            sorted_attn = torch.sort(spatial_attn, descending=True).values
            top_mass = sorted_attn[:top_k].sum() / spatial_mass
            coverage50 = float((sorted_attn.cumsum(0)
                                >= 0.5 * spatial_mass).nonzero(as_tuple=False)[0].item() + 1)
            coverage50 /= max(1, spatial_attn.numel())
            self._gg_attn_history.append(spatial_attn.detach())
            self._gg_attn_history = self._gg_attn_history[-self._gg_stability_window:]
            alpha_ok = alpha >= self._gg_onset_min_alpha
            concentration_ok = coverage50 <= self._gg_onset_max_coverage
            if alpha_ok and concentration_ok:
                self._gg_onset_hits += 1
            else:
                self._gg_onset_hits = 0
            onset_ready = self._gg_onset_hits >= self._gg_onset_patience
            fallback = li >= self._gg_max_onset_layer
            if onset_ready or fallback:
                self._gg_onset_layer = li
                should_prune = True
                self._gg_stages = {li: self._gg_keep_ratio}

        if should_prune:
            sp0, sp1 = self._gg_span
            stage_ratio = self._gg_stages.get(li, self._gg_keep_ratio)
            n_sp = sp1 - sp0
            fixed_selection = getattr(
                self, '_gg_fixed_spatial_deletion', None
            )
            if fixed_selection is not None:
                # Cross-layer replay: delete exactly this precomputed set,
                # defined in absolute (pre-pruning) coordinates.
                fixed = fixed_selection.to(
                    device=hidden_states.device, dtype=torch.long
                ).flatten()
                if fixed.numel() == 0:
                    raise ValueError('fixed spatial selection is empty')
                if int(fixed.min()) < sp0 or int(fixed.max()) >= sp1:
                    raise ValueError(
                        'fixed spatial selection is outside the spatial span'
                    )
                if torch.unique(fixed).numel() != fixed.numel():
                    raise ValueError(
                        'fixed spatial selection contains duplicate indices'
                    )
                keep = torch.ones(
                    n_sp,
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                keep[fixed - sp0] = False
                selected_spatial = torch.nonzero(keep).squeeze(-1)
            else:
                attn = layer_out[1][0].float().mean(dim=0)  # (T, T) head-avg
                scoring_fn = getattr(self, '_gg_scoring', None)
                scorer = getattr(self, '_gg_scorer', 'attn')
                if scoring_fn is not None:
                    scores = scoring_fn(
                        attn, sp0, sp1, hidden_states=hidden_states
                    )
                elif scorer == 'random':
                    scores = torch.rand(n_sp, device=attn.device)
                elif scorer == 'prompt_mean':
                    # mean of all prompt positions' attention to spatial (readout-aligned)
                    scores = attn[:-1, sp0:sp1].mean(dim=0)
                elif scorer == 'inverse_attn':
                    # prune the HIGH-attention anchors (causal reverse check)
                    scores = -attn[-1, sp0:sp1]
                elif scorer in ('stable_attn', 'stable_coverage') and self._gg_attn_history:
                    weights = torch.arange(
                        1, len(self._gg_attn_history) + 1,
                        device=attn.device, dtype=attn.dtype,
                    )
                    weights /= weights.sum()
                    scores = torch.stack(self._gg_attn_history).mul(
                        weights[:, None]).sum(dim=0)
                else:
                    scores = attn[-1, sp0:sp1]                   # last prompt token -> spatial
                if self._gg_geo_beta > 0 and self._gg_coords is not None:
                    from adaptive_token.geo_coords import diffuse_scores
                    scores = diffuse_scores(scores, self._gg_coords,
                                            beta=self._gg_geo_beta)
                if self._gg_prune_mode == 'delete_topk':
                    from adaptive_token.coverage_selector import keep_after_topk_deletion
                    selected_spatial = keep_after_topk_deletion(
                        scores, self._gg_delete_n
                    )
                else:
                    k = max(1, int(round(stage_ratio * n_sp)))
                    if scorer in ('coverage_attn', 'stable_coverage'):
                        from adaptive_token.coverage_selector import select_by_score_and_coverage
                        selected_spatial = select_by_score_and_coverage(
                            scores, self._gg_coords, k,
                            coverage_weight=getattr(self, '_gg_coverage_weight', 0.25),
                        )
                    else:
                        selected_spatial = torch.topk(
                            scores, min(k, n_sp)
                        ).indices
                    selected_spatial = torch.sort(selected_spatial).values
            keep_sp = selected_spatial + sp0
            dev = hidden_states.device
            cur_len = hidden_states.shape[1]
            G = self._gg_coarse_groups if len(self._gg_stages) == 1 else 0
            coarse_info = None
            if G > 0:
                # HiToken: pool DROPPED spatial tokens into G coarse tokens
                mask = torch.ones(n_sp, dtype=torch.bool, device=dev)
                mask[keep_sp - sp0] = False
                dropped = torch.nonzero(mask).squeeze(-1) + sp0   # ascending
                if dropped.numel() >= G:
                    chunks = None
                    if self._gg_geo_grouping and self._gg_coords is not None:
                        from adaptive_token.geo_coords import grid_groups
                        gg = grid_groups(dropped, self._gg_coords, sp0, G)
                        if gg is not None and len(gg) >= 2:
                            chunks = [c for c in gg if c.numel() > 0]
                    if chunks is None:
                        chunks = torch.chunk(dropped, G)
                    coarse_h = torch.stack(
                        [hidden_states[0, c].mean(0) for c in chunks])   # (G, D)
                    coarse_pos = torch.tensor(
                        [int(position_ids[0, c[len(c) // 2]].item()) for c in chunks],
                        device=dev)
                    coarse_info = (chunks, coarse_h, coarse_pos)
            if coarse_info is None:
                from adaptive_token.coverage_selector import merge_spatial_selection
                kept_idx_full = merge_spatial_selection(
                    sp0, sp1, cur_len, selected_spatial
                )
            else:
                kept_idx_full = torch.cat([
                    torch.arange(0, sp0, device=dev), keep_sp,
                    torch.arange(sp1, cur_len, device=dev)])
            if coarse_info is None:
                hidden_states = hidden_states[:, kept_idx_full, :]
                position_ids = position_ids[:, kept_idx_full]
                # bookkeeping for multi-stage: map current->original indices,
                # snapshot per cache length for the final per-layer KV prune
                self._gg_orig_map = self._gg_orig_map[kept_idx_full]
                self._gg_snapshots[hidden_states.shape[1]] = self._gg_orig_map.clone()
                # spatial span shrinks; coords follow the kept spatial subset
                self._gg_span = (sp0, sp0 + keep_sp.numel())
                if self._gg_coords is not None:
                    self._gg_coords = self._gg_coords[(keep_sp - sp0).cpu()]
            else:
                chunks, coarse_h, coarse_pos = coarse_info
                # merge fine + coarse inside the spatial span, ordered by position
                pre = torch.arange(0, sp0, device=dev)
                post = torch.arange(sp1, cur_len, device=dev)
                fine_pos = position_ids[0, keep_sp]
                sp_pos = torch.cat([fine_pos, coarse_pos])
                sp_vec = torch.cat([hidden_states[0, keep_sp], coarse_h.to(hidden_states.dtype)])
                order = torch.argsort(sp_pos)
                sp_pos, sp_vec = sp_pos[order], sp_vec[order]
                hidden_states = torch.cat([
                    hidden_states[0, pre], sp_vec, hidden_states[0, post]]).unsqueeze(0)
                position_ids = torch.cat([
                    position_ids[0, pre], sp_pos, position_ids[0, post]]).unsqueeze(0)
                # remember layout for KV pooling below
                self._gg_kv_plan = (pre, keep_sp, chunks, order, post)
            new_len = hidden_states.shape[1]
            # keep ORIGINAL positions (rotary cache expanded via _gg_maxpos)
            self._gg_next_pos = self._gg_maxpos - 640   # decode continues原始位置
            attn_mask_4d = _prepare_4d_causal_attention_mask(
                None, (batch, new_len), hidden_states, 0)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # prune/pool KV cache of ALL layers to the FINAL kept set for consistent decoding
    if prune_here and kept_idx_full is not None and use_cache:
        plan = getattr(self, '_gg_kv_plan', None)
        final_map = self._gg_orig_map            # original indices that survived
        final_len = final_map.shape[0]
        for i in range(len(past_key_values.key_cache)):
            K, V = past_key_values.key_cache[i], past_key_values.value_cache[i]
            L = K.shape[2]
            if plan is None:
                if L == final_len:
                    continue
                M = self._gg_snapshots.get(L)
                if M is None:
                    continue                      # unexpected length; leave as-is
                idx = snapshot_indices(M, final_map)
                past_key_values.key_cache[i] = K[:, :, idx, :]
                past_key_values.value_cache[i] = V[:, :, idx, :]
            else:
                pre, keep_sp, chunks, order, post = plan
                exp = pre.numel() + keep_sp.numel() + len(chunks) + post.numel()
                if L == exp:
                    continue
                ck = torch.stack([K[:, :, c, :].mean(2) for c in chunks], dim=2)
                cv = torch.stack([V[:, :, c, :].mean(2) for c in chunks], dim=2)
                sk = torch.cat([K[:, :, keep_sp, :], ck], dim=2)[:, :, order, :]
                sv = torch.cat([V[:, :, keep_sp, :], cv], dim=2)[:, :, order, :]
                past_key_values.key_cache[i] = torch.cat(
                    [K[:, :, pre, :], sk, K[:, :, post, :]], dim=2)
                past_key_values.value_cache[i] = torch.cat(
                    [V[:, :, pre, :], sv, V[:, :, post, :]], dim=2)
        self._gg_kv_plan = None
        self._gg_span = None  # consume span (one prefill per sample)
        self._gg_fixed_spatial_deletion = None

    next_cache = None
    if use_cache:
        next_cache = past_key_values.to_legacy_cache() if use_legacy_cache else past_key_values
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns]
                     if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states, past_key_values=next_cache,
        hidden_states=all_hidden_states, attentions=all_attns)
