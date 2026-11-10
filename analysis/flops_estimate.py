#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analytic FLOPs / KV-memory for single-drop token pruning on a Llama-7B-class LM.

Per-layer prefill FLOPs (batch 1, seq s, hidden d=4096, inter i=11008):
  attn: 4*s*d^2 (qkvo) + 2*s^2*d (scores+mix)
  mlp:  3*s*d*i(SwiGLU)         (constants cancel in ratios)
Decode step FLOPs per layer ~ 4*d^2 + 2*kv*d + 3*d*i  (kv = cached length).

Usage: python flops_estimate.py --n-spatial 1497 --n-text 220 \
        --prune-layer 14 --keep-ratio 0.10 [--decode-steps 8]
"""
import argparse

D, I, L = 4096, 11008, 32


def layer_prefill(s):
    return 4 * s * D * D + 2 * s * s * D + 3 * s * D * I


def layer_decode(kv):
    return 4 * D * D + 2 * kv * D + 3 * D * I


def estimate(n_spatial, n_text, prune_layer, keep_ratio, decode_steps=8):
    s_full = n_spatial + n_text
    s_red = int(round(n_spatial * keep_ratio)) + n_text
    full_blocks = prune_layer + 1
    reduced_blocks = L - full_blocks

    base_prefill = L * layer_prefill(s_full)
    ours_prefill = (full_blocks * layer_prefill(s_full)
                    + reduced_blocks * layer_prefill(s_red))
    base_decode = decode_steps * L * layer_decode(s_full)
    ours_decode = decode_steps * L * layer_decode(s_red)
    base = base_prefill + base_decode
    ours = ours_prefill + ours_decode
    return {
        'seq_full': s_full,
        'seq_reduced': s_red,
        'full_blocks': full_blocks,
        'reduced_blocks': reduced_blocks,
        'prefill_ratio': ours_prefill / base_prefill,
        'decode_ratio': ours_decode / base_decode,
        'total_ratio': ours / base,
        'kv_ratio': s_red / s_full,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-spatial', type=int, default=1497)
    ap.add_argument('--n-text', type=int, default=220)
    ap.add_argument('--prune-layer', type=int, required=True)
    ap.add_argument('--keep-ratio', type=float, required=True)
    ap.add_argument('--decode-steps', type=int, default=8)
    a = ap.parse_args()

    stats = estimate(
        a.n_spatial, a.n_text, a.prune_layer, a.keep_ratio, a.decode_steps
    )
    print(f"seq full={stats['seq_full']} reduced={stats['seq_reduced']}")
    print(f"blocks full={stats['full_blocks']} reduced={stats['reduced_blocks']}")
    print(f"prefill FLOPs ratio = {stats['prefill_ratio']:.3f}")
    print(f"decode  FLOPs ratio = {stats['decode_ratio']:.3f}")
    print(
        f"total   FLOPs ratio = {stats['total_ratio']:.3f}  "
        f"(-{(1 - stats['total_ratio']) * 100:.1f}%)"
    )
    print(
        f"decode KV memory    = {stats['kv_ratio']:.3f}  "
        f"(-{(1 - stats['kv_ratio']) * 100:.1f}%)"
    )


if __name__ == '__main__':
    main()
