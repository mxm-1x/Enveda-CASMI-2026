#!/usr/bin/env python3
"""Rank an external structure table by precursor exact-mass agreement."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrieve import neutral_mass


def retrieve_candidates(
    candidate_path: str,
    query_path: str,
    *,
    top_k: int = 25,
    window_ppm: float = 1_000.0,
) -> pd.DataFrame:
    candidates = pd.read_parquet(
        candidate_path,
        columns=["smiles", "metric_key", "exact_mass"],
    )
    required = {"smiles", "metric_key", "exact_mass"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"Candidate table must contain {sorted(required)}")
    candidates = candidates.dropna(subset=["smiles", "metric_key", "exact_mass"])
    candidates["exact_mass"] = pd.to_numeric(candidates["exact_mass"], errors="coerce")
    candidates = candidates[np.isfinite(candidates["exact_mass"]) & (candidates["exact_mass"] > 0)]
    if candidates["metric_key"].duplicated().any():
        raise ValueError("Candidate table must be deduplicated by metric_key")
    candidates = candidates.sort_values("exact_mass", kind="mergesort").reset_index(drop=True)
    candidate_masses = candidates["exact_mass"].to_numpy(dtype=np.float64)
    candidate_smiles = candidates["smiles"].to_numpy()

    queries = pd.read_parquet(
        query_path,
        columns=["molecule_id", "precursor_mz", "adduct", "ionization_mode"],
    )
    rows = []
    no_precursor = 0
    for molecule_id, group in queries.groupby("molecule_id", sort=False):
        query_masses = []
        for row in group.itertuples(index=False):
            mass = neutral_mass(row.precursor_mz, row.adduct)
            if mass is not None and np.isfinite(mass) and mass > 0:
                query_masses.append(float(mass))
        if not query_masses:
            no_precursor += 1
            raise ValueError(f"No recognized precursor/adduct mass for {molecule_id}")

        mass_center = float(np.median(query_masses))
        radius = max(mass_center * window_ppm * 1e-6, 0.01)
        left = int(np.searchsorted(candidate_masses, mass_center - radius, side="left"))
        right = int(np.searchsorted(candidate_masses, mass_center + radius, side="right"))
        candidate_indices = np.arange(left, right, dtype=np.int64)

        # Keep up to 25 nearest-mass candidates outside the window as a fallback
        # if the structure database contains fewer than 25 matching masses.
        if len(candidate_indices) < top_k:
            nearest_right = int(np.searchsorted(candidate_masses, mass_center))
            nearest_left = nearest_right - 1
            selected = set(candidate_indices.tolist())
            target_count = min(top_k, len(candidate_masses))
            while len(selected) < target_count:
                if nearest_left < 0:
                    selected.add(nearest_right)
                    nearest_right += 1
                elif nearest_right >= len(candidate_masses):
                    selected.add(nearest_left)
                    nearest_left -= 1
                elif abs(candidate_masses[nearest_left] - mass_center) <= abs(candidate_masses[nearest_right] - mass_center):
                    selected.add(nearest_left)
                    nearest_left -= 1
                else:
                    selected.add(nearest_right)
                    nearest_right += 1
            candidate_indices = np.asarray(sorted(selected), dtype=np.int64)

        observed = np.asarray(query_masses, dtype=np.float64)
        errors = np.abs(candidate_masses[candidate_indices, None] - observed[None, :])
        ppm_errors = errors / observed[None, :] * 1e6
        median_error = np.median(ppm_errors, axis=1)
        mean_error = np.mean(ppm_errors, axis=1)
        order = np.lexsort((candidates.iloc[candidate_indices]["metric_key"].to_numpy(), mean_error, median_error))
        ranked_indices = candidate_indices[order[:top_k]]
        rows.append({
            "molecule_id": molecule_id,
            "smiles": ";".join(candidate_smiles[ranked_indices]),
        })

    print(
        f"Ranked {len(rows):,} molecules against {len(candidates):,} candidate structures; "
        f"candidate window={window_ppm:g} ppm; missing precursor groups={no_precursor}"
    )
    return pd.DataFrame(rows, columns=["molecule_id", "smiles"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", default="candidate_predictions.csv")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--window-ppm", type=float, default=1_000.0)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 25:
        parser.error("--top-k must be between 1 and 25")
    if args.window_ppm <= 0:
        parser.error("--window-ppm must be positive")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = retrieve_candidates(
        args.candidates,
        args.queries,
        top_k=args.top_k,
        window_ppm=args.window_ppm,
    )
    predictions.to_csv(output_path, index=False)
    print(f"Wrote {len(predictions):,} molecule predictions to {output_path}")


if __name__ == "__main__":
    main()
