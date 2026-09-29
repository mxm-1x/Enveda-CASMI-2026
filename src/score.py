"""Competition metric helpers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_at_25(ranked_keys: Sequence[str], truth_key: str) -> float:
    """Reciprocal rank of the first correct connectivity key, capped at 25."""
    if not truth_key:
        return 0.0
    for rank, key in enumerate(ranked_keys[:25], start=1):
        if key == truth_key:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank_at_25(
    predictions: Mapping[str, Sequence[str]], truths: Mapping[str, str]
) -> float:
    """Average molecule-level MRR@25 over all truth IDs."""
    if not truths:
        return 0.0
    return sum(
        reciprocal_rank_at_25(predictions.get(molecule_id, ()), truth)
        for molecule_id, truth in truths.items()
    ) / len(truths)
