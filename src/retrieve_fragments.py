#!/usr/bin/env python3
"""Rerank exact-mass candidates by simple one-bond fragment evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrieve import neutral_mass

PROTON = 1.007276


def _candidate_chemistry(smiles: str, precursor_mass: float) -> tuple[np.ndarray, dict[str, int]]:
    """Predict simple fragments and compact structural descriptors."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import Descriptors
    from rdkit.Chem import Lipinski, rdMolDescriptors

    rdBase.DisableLog("rdApp.warning")
    rdBase.DisableLog("rdApp.error")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.empty(0, dtype=np.float64), {
            "heavy_atom_count": 0, "hetero_atom_count": 0, "ring_count": 0,
            "aromatic_ring_count": 0, "rotatable_bond_count": 0,
        }

    descriptors = {
        "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        "hetero_atom_count": int(sum(atom.GetAtomicNum() not in (1, 6) for atom in molecule.GetAtoms())),
        "ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "rotatable_bond_count": int(Lipinski.NumRotatableBonds(molecule)),
    }

    masses: set[float] = set()
    for bond in molecule.GetBonds():
        # Ring cuts remain connected and aromatic cuts are not represented by
        # this simple neutral-fragment model. Avoid spending time on them.
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        try:
            cleaved = Chem.FragmentOnBonds(molecule, [bond.GetIdx()], addDummies=False)
            fragments = Chem.GetMolFrags(cleaved, asMols=True, sanitizeFrags=True)
        except (RuntimeError, ValueError):
            continue
        for fragment in fragments:
            if fragment.GetNumHeavyAtoms() < 3:
                continue
            mass = float(Descriptors.ExactMolWt(fragment))
            # Exclude intact-parent and nearly intact ring-opened products.
            if 0 < mass < precursor_mass - 10.0:
                masses.add(round(mass, 6))
    return np.asarray(sorted(masses), dtype=np.float64), descriptors


def _evidence_by_mode(group: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    evidence: dict[str, dict[float, float]] = {"positive": {}, "negative": {}}
    for row in group.itertuples(index=False):
        mode = str(row.ionization_mode).strip().lower()
        if mode not in evidence:
            continue
        if not isinstance(row.ms2_mzs, (list, tuple, np.ndarray)) or not isinstance(
            row.ms2_normalized_intensities, (list, tuple, np.ndarray)
        ):
            continue
        for mz, intensity in zip(row.ms2_mzs, row.ms2_normalized_intensities):
            try:
                mz, intensity = float(mz), float(intensity)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(mz) or not np.isfinite(intensity) or mz <= 0 or intensity <= 0:
                continue
            # Retain the strongest observation for each nominal 0.01 Da bin
            # across collision energies and replicate spectra.
            bin_mz = round(mz, 2)
            evidence[mode][bin_mz] = max(evidence[mode].get(bin_mz, 0.0), intensity)
    return {
        mode: (
            np.asarray(sorted(peaks), dtype=np.float64),
            np.asarray([peaks[mz] for mz in sorted(peaks)], dtype=np.float64),
        )
        for mode, peaks in evidence.items()
    }


def _matched_intensity(target: float, peaks: np.ndarray, intensities: np.ndarray, tolerance_da: float) -> float:
    if not len(peaks):
        return 0.0
    index = int(np.searchsorted(peaks, target))
    best = 0.0
    for candidate_index in (index - 1, index):
        if 0 <= candidate_index < len(peaks) and abs(peaks[candidate_index] - target) <= tolerance_da:
            best = max(best, float(intensities[candidate_index]))
    return best


def rerank_candidates(
    candidate_path: str,
    query_path: str,
    *,
    top_k: int = 25,
    window_ppm: float = 10.0,
    fragment_tolerance_da: float = 0.02,
    features_output: str | None = None,
) -> pd.DataFrame:
    candidates = pd.read_parquet(
        candidate_path, columns=["smiles", "metric_key", "exact_mass"]
    ).dropna(subset=["smiles", "metric_key", "exact_mass"])
    candidates["exact_mass"] = pd.to_numeric(candidates["exact_mass"], errors="coerce")
    candidates = candidates[np.isfinite(candidates["exact_mass"]) & (candidates["exact_mass"] > 0)]
    if candidates["metric_key"].duplicated().any():
        raise ValueError("Candidate table must be deduplicated by metric_key")
    candidates = candidates.sort_values("exact_mass", kind="mergesort").reset_index(drop=True)
    candidate_masses = candidates["exact_mass"].to_numpy(dtype=np.float64)
    query = pd.read_parquet(query_path)
    required = {
        "molecule_id", "precursor_mz", "adduct", "ionization_mode",
        "ms2_mzs", "ms2_normalized_intensities",
    }
    if not required.issubset(query.columns):
        raise ValueError(f"Query table missing columns: {sorted(required - set(query.columns))}")

    fragment_cache: dict[str, tuple[np.ndarray, dict[str, int]]] = {}
    feature_rows = []
    rows = []
    query_groups = list(query.groupby("molecule_id", sort=False))
    for group_number, (molecule_id, group) in enumerate(query_groups, 1):
        masses = [neutral_mass(mz, adduct) for mz, adduct in zip(group.precursor_mz, group.adduct)]
        observed_masses = [mass for mass in masses if mass is not None and np.isfinite(mass) and mass > 0]
        if not observed_masses:
            raise ValueError(f"No recognized precursor/adduct mass for {molecule_id}")
        center = float(np.median(observed_masses))
        radius = max(center * window_ppm * 1e-6, 0.01)
        left = int(np.searchsorted(candidate_masses, center - radius, side="left"))
        right = int(np.searchsorted(candidate_masses, center + radius, side="right"))
        indices = np.arange(left, right, dtype=np.int64)
        if not len(indices):
            # Keep a chemically relevant fallback if calibration puts every
            # candidate outside the narrow high-recall precursor window.
            nearest = int(np.searchsorted(candidate_masses, center))
            indices = np.arange(max(0, nearest - top_k // 2), min(len(candidates), nearest + top_k // 2 + 1))

        evidence = _evidence_by_mode(group)
        scored = []
        for index in indices:
            candidate = candidates.iloc[int(index)]
            key = str(candidate.metric_key)
            chemistry = fragment_cache.get(key)
            if chemistry is None:
                chemistry = _candidate_chemistry(str(candidate.smiles), float(candidate.exact_mass))
                fragment_cache[key] = chemistry
            fragments, descriptors = chemistry

            intensity_sum = 0.0
            predicted_count = 0
            matched_count = 0
            for neutral_fragment_mass in fragments:
                matched_fragment = False
                for mode, ion_mass in (
                    ("positive", neutral_fragment_mass + PROTON),
                    ("negative", neutral_fragment_mass - PROTON),
                ):
                    peak_mzs, peak_intensities = evidence[mode]
                    if not len(peak_mzs):
                        continue
                    predicted_count += 1
                    matched_intensity = _matched_intensity(
                        float(ion_mass), peak_mzs, peak_intensities, fragment_tolerance_da
                    )
                    intensity_sum += matched_intensity
                    matched_fragment |= matched_intensity > 0
                matched_count += int(matched_fragment)
            score = intensity_sum / np.sqrt(max(predicted_count, 1))
            mass_error_ppm = abs(float(candidate.exact_mass) - center) / center * 1e6
            scored.append((score, mass_error_ppm, key, str(candidate.smiles)))
            feature_rows.append({
                "molecule_id": str(molecule_id),
                "metric_key": key,
                "smiles": str(candidate.smiles),
                "mass_error_ppm": mass_error_ppm,
                "fragment_score": score,
                "fragment_intensity_sum": intensity_sum,
                "matched_fragment_count": matched_count,
                "predicted_fragment_count": predicted_count,
                **descriptors,
            })

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        rows.append({"molecule_id": molecule_id, "smiles": ";".join(item[3] for item in scored[:top_k])})
        if group_number % 25 == 0 or group_number == len(query_groups):
            print(
                f"Scored {group_number:,}/{len(query_groups):,} molecules; "
                f"cached fragments for {len(fragment_cache):,} candidate structures",
                flush=True,
            )

    print(
        f"Fragment-reranked {len(rows):,} molecules; candidates in {window_ppm:g} ppm "
        f"window; generated fragment sets for {len(fragment_cache):,} unique structures"
    )
    if features_output:
        feature_path = Path(features_output)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(feature_rows).to_parquet(feature_path, index=False)
        print(f"Wrote {len(feature_rows):,} candidate feature rows to {feature_path}")
    return pd.DataFrame(rows, columns=["molecule_id", "smiles"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", default="fragment_predictions.csv")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--window-ppm", type=float, default=10.0)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    parser.add_argument("--features-output", help="Optional parquet with per-query candidate ranking features")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 25:
        parser.error("--top-k must be between 1 and 25")
    if args.window_ppm <= 0 or args.fragment_tolerance_da <= 0:
        parser.error("Mass window and fragment tolerance must be positive")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = rerank_candidates(
        args.candidates,
        args.queries,
        top_k=args.top_k,
        window_ppm=args.window_ppm,
        fragment_tolerance_da=args.fragment_tolerance_da,
        features_output=args.features_output,
    )
    predictions.to_csv(output_path, index=False)
    print(f"Wrote {len(predictions):,} molecule predictions to {output_path}")


if __name__ == "__main__":
    main()
