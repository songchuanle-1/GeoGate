# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# LLaVA-3D integration; see THIRD_PARTY.md for upstream attribution and terms.
"""Per-token 3D coordinates for GeoGate: hook the RGBDVideoTower so that every
prefill also records the voxel-pooled 3D centroid of each spatial token
(identical scatter_mean/voxelize path as the token features themselves).

Also provides the geo utilities used by geogate:
  * diffuse_scores: 1-step kNN propagation of attention scores over 3D graph
  * grid_groups:    coarse voxel-grid grouping of dropped tokens (geo-aware
                    HiToken, replaces naive contiguous chunks)
"""
import types

import torch
from torch_scatter import scatter_mean


def pool_voxel_coordinates(feat_xyz, p2v):
    """Pool per-view coordinates with the exact token-to-voxel mapping."""
    if feat_xyz.dim() != 5:
        raise ValueError("feat_xyz must have shape (B, V, H, W, 3)")
    if p2v.dim() != 2 or p2v.shape[0] != feat_xyz.shape[0]:
        raise ValueError("p2v must have shape (B, V*H*W)")
    xyz = feat_xyz.reshape(feat_xyz.shape[0], -1, 3)
    if p2v.shape[1] != xyz.shape[1]:
        raise ValueError("p2v point count does not match feat_xyz")
    return torch.cat(
        [scatter_mean(xyz[b], p2v[b], dim=0) for b in range(p2v.shape[0])]
    ).float()


def attach_coord_hook(model):
    """Wrap video_tower.forward to stash per-token 3D coords on model.model."""
    lm = model.model
    tower = model.get_video_tower()
    if getattr(tower, '_gg_coord_hooked', False):
        return model
    orig_forward = tower.forward

    def forward(self, features, depths, poses, intrinsics, lengths=None):
        out = orig_forward(features, depths, poses, intrinsics, lengths)
        try:
            from llava.model.multimodal_encoder.unproject import (
                backprojector_dataloader, voxelize)
            feat_xyz, _ = backprojector_dataloader(
                [features.flatten(0, 1)], depths, poses, intrinsics)
            p2v = voxelize(feat_xyz, self.voxel_size)          # (B, N)
            lm._gg_coords = pool_voxel_coordinates(feat_xyz, p2v)
        except Exception as e:  # never break inference for coords
            lm._gg_coords = None
            print(f'[geo_coords] hook failed: {e}')
        return out

    tower.forward = types.MethodType(forward, tower)
    tower._gg_coord_hooked = True
    return model


def attach_ost_coord_hook(model):
    """3D-LLaVA: wrap the OST segmentor.forward to stash per-token 3D coords
    (topk_query_coord, one centroid per spatial token) on model.model._gg_coords.
    The superpoint architecture exposes coordinates directly from the visual
    sampler, unlike LLaVA-3D's voxel re-derivation."""
    lm = model.model
    tower = model.get_pointcloud_tower()
    if getattr(tower, '_gg_ost_hooked', False):
        return model
    seg = tower.segmentor

    def hook(module, inp, out):
        # out = (topk_query_feat, topk_query_coord, aligned_sp_feat, x, sp_xyz, ...)
        try:
            lm._gg_coords = out[1][0].float()      # (num_tokens, 3), batch=1
        except Exception as e:
            lm._gg_coords = None
            print(f'[geo_coords] OST hook failed: {e}')

    seg.register_forward_hook(hook)
    tower._gg_ost_hooked = True
    return model


def diffuse_scores(scores, coords, k=8, beta=0.5):
    """score' = norm(score) + beta * norm(mean score of k 3D-nearest neighbors)."""
    n = scores.shape[0]
    if coords is None or coords.shape[0] != n or n <= k:
        return scores
    c = coords.to(scores.device, dtype=scores.dtype)
    dist = torch.cdist(c, c)                                   # (N, N)
    knn = dist.topk(k + 1, largest=False).indices[:, 1:]       # exclude self
    neigh = scores[knn].mean(dim=-1)

    def _n(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    return _n(scores) + beta * _n(neigh)


def grid_groups(dropped_idx, coords, sp0, target_groups):
    """Group dropped spatial tokens by coarse 3D voxel cells (geo-aware HiToken).

    Returns a list of index tensors (subsets of dropped_idx). Cell size is
    chosen from the scene extent so the group count lands near target_groups.
    """
    if coords is None:
        return None
    local = dropped_idx - sp0
    if local.max() >= coords.shape[0]:
        return None
    c = coords[local.cpu()].to(dropped_idx.device)
    extent = (c.max(0).values - c.min(0).values).clamp(min=1e-3)
    cell = (extent.prod() / max(target_groups, 1)).pow(1 / 3).clamp(min=1e-3)
    cid = ((c - c.min(0).values) / cell).floor().long()
    key = cid[:, 0] * 73856093 ^ cid[:, 1] * 19349663 ^ cid[:, 2] * 83492791
    uniq, inv = torch.unique(key, return_inverse=True)
    return [dropped_idx[inv == g] for g in range(len(uniq))]
