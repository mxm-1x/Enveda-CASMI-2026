"""Small, dependency-light MS/MS spectrum preprocessing and similarity."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping


def peak_vector(
    mzs: Iterable[float],
    intensities: Iterable[float],
    *,
    bin_width: float = 0.02,
    min_relative_intensity: float = 0.005,
    max_peaks: int = 256,
    precursor_mz: float | None = None,
) -> dict[int, float]:
    """Clean peaks and return an L2-normalized sparse binned vector.

    Bin IDs are integer, avoiding Python float-key rounding differences. The
    precursor cut is conservative: peaks up to precursor + 1 Da are retained
    to tolerate isotopes and source-specific preprocessing.
    """
    pairs = []
    base = 0.0
    for mz, intensity in zip(mzs, intensities):
        try:
            mz, intensity = float(mz), float(intensity)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(mz) or not math.isfinite(intensity) or mz <= 0 or intensity <= 0:
            continue
        if precursor_mz is not None and mz > float(precursor_mz) + 1.0:
            continue
        pairs.append((mz, intensity))
        base = max(base, intensity)

    if not pairs or base <= 0:
        return {}
    floor = base * min_relative_intensity
    pairs = [(mz, inten) for mz, inten in pairs if inten >= floor]
    if max_peaks > 0 and len(pairs) > max_peaks:
        pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)[:max_peaks]

    bins: dict[int, float] = defaultdict(float)
    for mz, intensity in pairs:
        bins[int(round(mz / bin_width))] += math.sqrt(intensity / base)
    norm = math.sqrt(sum(value * value for value in bins.values()))
    if not norm:
        return {}
    return {key: value / norm for key, value in bins.items()}


def cosine(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    """Cosine similarity for normalized sparse vectors."""
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def neutral_loss_vector(
    precursor_mz: float,
    mzs: Iterable[float],
    intensities: Iterable[float],
    *,
    bin_width: float = 0.02,
    max_peaks: int = 256,
) -> dict[int, float]:
    """Represent precursor-to-fragment neutral losses as another sparse vector."""
    losses = []
    weights = []
    for mz, intensity in zip(mzs, intensities):
        try:
            loss = float(precursor_mz) - float(mz)
            intensity = float(intensity)
        except (TypeError, ValueError):
            continue
        if loss > 0 and intensity > 0:
            losses.append(loss)
            weights.append(intensity)
    return peak_vector(
        losses,
        weights,
        bin_width=bin_width,
        min_relative_intensity=0.0,
        max_peaks=max_peaks,
    )
