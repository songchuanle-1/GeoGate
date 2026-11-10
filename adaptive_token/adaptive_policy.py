# SPDX-License-Identifier: Apache-2.0
"""Small, deterministic policies used by Adaptive GeoGate."""

import re


COUNT_PATTERNS = (
    r"\bhow many\b",
    r"\bnumber of\b",
    r"\bcount\b",
    r"\bhow much\b",
)
SPATIAL_PATTERNS = (
    r"\bwhere\b",
    r"\bleft\b",
    r"\bright\b",
    r"\babove\b",
    r"\bbelow\b",
    r"\bbehind\b",
    r"\bin front of\b",
    r"\bnext to\b",
    r"\bbetween\b",
    r"\bcorner\b",
    r"\bside\b",
    r"\bdirection\b",
    r"\bfacing\b",
)


def question_keep_ratio(question, base_ratio=0.10, min_ratio=None, max_ratio=0.50):
    """Return a deterministic instruction-conditioned spatial-token budget.

    The policy is intentionally simple and interpretable. Counting questions
    receive the largest budget, followed by relational questions; all other
    questions retain the base ratio.
    """
    text = question.lower()
    min_ratio = base_ratio if min_ratio is None else min_ratio
    ratio = max(base_ratio, min_ratio)
    if any(re.search(pattern, text) for pattern in COUNT_PATTERNS):
        ratio *= 2.5
    elif any(re.search(pattern, text) for pattern in SPATIAL_PATTERNS):
        ratio *= 2.0
    return min(float(max_ratio), float(ratio))
