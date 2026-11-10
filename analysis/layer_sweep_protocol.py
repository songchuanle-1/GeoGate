# SPDX-License-Identifier: Apache-2.0
"""Pure protocol helpers for the paired measured-layer sweep."""

from __future__ import annotations


DEFAULT_LAYERS = (2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 18)


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain unique positive integers")
    if tuple(sorted(layers)) != layers:
        raise ValueError("--layers must be strictly increasing")
    if layers[0] < 0 or layers[-1] > 31:
        raise ValueError("--layers must lie in [0, 31]")
    return layers


def grouped_rows(rows: list[dict], conditions: tuple[str, ...]) -> dict:
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        condition = str(row["condition"])
        if condition not in conditions:
            raise ValueError(f"Unexpected condition: {condition}")
        group = grouped.setdefault(question_id, {})
        if condition in group:
            raise ValueError(f"Duplicate row: {question_id}/{condition}")
        group[condition] = row
    expected = set(conditions)
    for question_id, group in grouped.items():
        if set(group) != expected:
            raise ValueError(f"Incomplete group: {question_id}")
    return grouped
