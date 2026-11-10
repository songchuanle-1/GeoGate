# SPDX-License-Identifier: Apache-2.0
"""Configure one frozen stage arm without re-installing GeoGate hooks."""

from analysis.stage_protocol_v4 import arms, token_counts


def configure_arm(model, arm, spatial_start, spatial_count):
    if arm not in arms(arm["mode"]):
        raise ValueError("Arm differs from frozen protocol")
    if spatial_start < 0:
        raise ValueError("Negative spatial start")
    from adaptive_token.geogate_v0 import set_keep_ratio, set_span
    lm = model.model
    if getattr(lm, "_gg_stages_user_set", False):
        raise ValueError("Multistage configuration is outside this protocol")
    counts = token_counts(spatial_count, arm["mode"])
    lm._gg_prune_layer = arm["layer"]
    lm._gg_scorer = arm["scorer"]
    lm._gg_prune_mode = arm["prune_mode"]
    lm._gg_delete_n = counts["delete"] if arm["mode"] == "delete10" else 0
    lm._gg_scoring = None
    lm._gg_adaptive_onset = False
    set_keep_ratio(model, counts["keep"] / spatial_count)
    set_span(model, spatial_start, spatial_start + spatial_count)
    return counts
