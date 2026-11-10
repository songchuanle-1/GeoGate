# SPDX-License-Identifier: Apache-2.0
"""Frozen pure-Python protocol helpers for the next GPU follow-up."""

import hashlib


SUFFIX = "Answer the question simply."
CONDITIONS = ("original", "no_suffix", "paraphrase")


def prompt_variants(text):
    text = text.strip()
    if not text.endswith(SUFFIX):
        raise ValueError("Original prompt suffix is missing; do not guess a rewrite")
    question = text[:-len(SUFFIX)].rstrip()
    if not question:
        raise ValueError("Question is empty")
    return {
        "original": text,
        "no_suffix": question,
        "paraphrase": question + " Give a short answer.",
    }


def sample_seed(question_id, seed=20260908):
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def onset(frac, coverage, first=10, last=18, alpha=0.10, maximum=0.12, patience=2):
    hits = 0
    for layer in range(first, last + 1):
        hits = hits + 1 if frac[layer] >= alpha and coverage[layer] <= maximum else 0
        if hits >= patience:
            return {"layer": layer, "fallback": False}
    return {"layer": last, "fallback": True}


def rotation(items, index):
    offset = index % len(items)
    return items[offset:] + items[:offset]
