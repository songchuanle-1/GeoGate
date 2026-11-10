# SPDX-License-Identifier: Apache-2.0
"""Frozen stage-intervention arms for 3D-LLaVA; not a GPU launcher."""

LAYERS = (8, 14, 20)
SEEDS = (101, 202, 303)
RANKINGS = ("top", "random", "bottom")


def arms(mode):
    if mode not in ("delete10", "retain10"):
        raise ValueError("Unknown intervention")
    return [
        {"name": f"{mode}_L{layer}_{ranking}_s{seed}", "layer": layer,
         "ranking": ranking, "seed": seed, "mode": mode,
         "scorer": {"top": "attn", "random": "random",
                    "bottom": "inverse_attn"}[ranking],
         "prune_mode": "delete_topk" if mode == "delete10" else "keep_topk"}
        for layer in LAYERS for ranking in RANKINGS for seed in SEEDS
    ]


def token_counts(spatial, mode):
    if not isinstance(spatial, int) or spatial < 1:
        raise ValueError("Positive integer spatial count required")
    if mode == "retain10":
        keep = max(1, round(0.1 * spatial))
    elif mode == "delete10":
        keep = spatial - min(spatial - 1, max(1, round(0.1 * spatial)))
    else:
        raise ValueError("Unknown intervention")
    return {"keep": keep, "delete": spatial - keep}


def stage_contrasts(scores):
    """Scores keyed by (layer, ranking, seed), higher is better, one question.

    Random-minus-top deletion quality measures extra damage from deleting top
    tokens. Native cancels in this contrast. Average seeds before bootstrap.
    """
    expected = {(layer, rank, seed) for layer in LAYERS
                for rank in ("top", "random") for seed in SEEDS}
    if not expected.issubset(scores):
        raise ValueError("Incomplete primary paired deletion scores")
    damage = {layer: sum(scores[layer, "random", seed] -
                         scores[layer, "top", seed] for seed in SEEDS) / len(SEEDS)
              for layer in LAYERS}
    return {"L14_minus_L8": damage[14] - damage[8],
            "L14_minus_L20": damage[14] - damage[20]}
