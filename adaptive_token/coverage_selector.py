# SPDX-License-Identifier: Apache-2.0
"""Geometry-aware token selection utilities."""

import torch


def merge_spatial_selection(sp0, sp1, current_length, spatial_indices):
    """Keep all non-spatial tokens and replace the spatial span with selections."""
    pre = torch.arange(0, sp0, device=spatial_indices.device)
    post = torch.arange(sp1, current_length, device=spatial_indices.device)
    kept_spatial = torch.sort(spatial_indices).values + sp0
    return torch.cat([pre, kept_spatial, post])


def keep_after_topk_deletion(scores, delete_n):
    """Return local indices kept after deleting the largest `delete_n` scores."""
    n = int(scores.numel())
    delete_n = min(max(0, int(delete_n)), max(0, n - 1))
    if delete_n == 0:
        return torch.arange(n, device=scores.device)
    deleted = torch.topk(scores, delete_n).indices
    keep_mask = torch.ones(n, dtype=torch.bool, device=scores.device)
    keep_mask[deleted] = False
    return torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)


def select_by_score_and_coverage(scores, coords, k, coverage_weight=0.25):
    """Select high-score tokens while discouraging spatial duplicates.

    The first token is the highest-scoring token. Subsequent tokens maximize a
    convex combination of normalized score and distance to the selected set.
    The function returns local indices and falls back to top-k if coordinates
    are unavailable or malformed.
    """
    n = int(scores.numel())
    k = max(1, min(int(k), n))
    if coords is None or coords.ndim != 2 or coords.shape[0] != n or coords.shape[1] != 3:
        return torch.topk(scores, k).indices
    if not torch.isfinite(coords).all():
        return torch.topk(scores, k).indices

    score = scores.float()
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    coord = coords.to(score.device, dtype=score.dtype)
    extent = (coord.max(dim=0).values - coord.min(dim=0).values).norm().clamp_min(1e-6)
    selected = [int(torch.argmax(score).item())]
    available = torch.ones(n, dtype=torch.bool, device=score.device)
    available[selected[0]] = False
    min_dist = torch.cdist(coord, coord[selected[0]:selected[0] + 1]).squeeze(1) / extent

    weight = float(max(0.0, min(1.0, coverage_weight)))
    for _ in range(1, k):
        utility = (1.0 - weight) * score + weight * min_dist
        utility[~available] = -float("inf")
        chosen = int(torch.argmax(utility).item())
        selected.append(chosen)
        available[chosen] = False
        dist = torch.cdist(coord, coord[chosen:chosen + 1]).squeeze(1) / extent
        min_dist = torch.minimum(min_dist, dist)
    return torch.tensor(selected, dtype=torch.long, device=scores.device)
