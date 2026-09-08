"""Reusable drift metrics for production-style model monitoring."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


_EPS = 1e-8


def _safe_ratio(value: float, total: float) -> float:
    return max(value / total, _EPS)


def population_stability_index_numeric(
    reference: Sequence[float],
    current: Sequence[float],
    bins: int = 10,
) -> float:
    """Compute PSI for numeric distributions using reference-derived bins."""
    if not reference or not current:
        raise ValueError("reference and current must both be non-empty")
    if bins < 2:
        raise ValueError("bins must be >= 2")

    ref_sorted = sorted(float(v) for v in reference)
    n = len(ref_sorted)
    edges = []
    for i in range(1, bins):
        idx = min(n - 1, max(0, round(i * (n - 1) / bins)))
        edges.append(ref_sorted[idx])

    # Remove duplicate boundaries so constant / low-cardinality inputs remain valid.
    edges = sorted(set(edges))

    def bucket(value: float) -> int:
        for idx, edge in enumerate(edges):
            if value <= edge:
                return idx
        return len(edges)

    bucket_count = len(edges) + 1
    ref_counts = [0] * bucket_count
    cur_counts = [0] * bucket_count

    for value in reference:
        ref_counts[bucket(float(value))] += 1
    for value in current:
        cur_counts[bucket(float(value))] += 1

    psi = 0.0
    for ref_count, cur_count in zip(ref_counts, cur_counts):
        ref_pct = _safe_ratio(ref_count, len(reference))
        cur_pct = _safe_ratio(cur_count, len(current))
        psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
    return psi


def population_stability_index_categorical(
    reference: Iterable[str],
    current: Iterable[str],
) -> float:
    """Compute PSI for categorical distributions."""
    ref = [str(v) for v in reference]
    cur = [str(v) for v in current]
    if not ref or not cur:
        raise ValueError("reference and current must both be non-empty")

    ref_counts = Counter(ref)
    cur_counts = Counter(cur)
    categories = sorted(set(ref_counts) | set(cur_counts))

    psi = 0.0
    for category in categories:
        ref_pct = _safe_ratio(ref_counts.get(category, 0), len(ref))
        cur_pct = _safe_ratio(cur_counts.get(category, 0), len(cur))
        psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
    return psi


def drift_status(score: float) -> str:
    """Map PSI to a human-readable monitoring status."""
    if score >= 0.25:
        return "alert"
    if score >= 0.10:
        return "watch"
    return "stable"
