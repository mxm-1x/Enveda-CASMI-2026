#!/usr/bin/env python3
"""Rank mass-compatible structures by evidence transferred from spectral analogs.

For each query molecule, this streams the reference MS/MS library once, finds
spectra with compatible polarity and a nearby (but not necessarily equal)
neutral precursor mass, and scores candidate structures by
max(similarity**power * Morgan-Tanimoto(candidate, analog)). This is an
independent feature channel; it does not change the existing spectral ranking.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

from chemistry import metric_inchikey14
from retrieve import neutral_mass, _text
from spectra import peak_vector

TRAIN_COLUMNS = [
    "normalized_smiles", "inchikey14", "ionization_mode", "adduct",
    "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
]
QUERY_COLUMNS = [
    "molecule_id", "ionization_mode", "adduct", "precursor_mz",
    "ms2_mzs", "ms2_normalized_intensities",
]
FINGERPRINT_BITS = 2048
MASS_BINS = 200_000  # 4,000 m/z at 0.02 Da/bin


def _metric_key(smiles: str, source_key: str, cache: dict[str, str]) -> str:
    cache_key = source_key or smiles
    if cache_key not in cache:
        try:
            cache[cache_key] = metric_inchikey14(smiles)
        except Exception:
            cache[cache_key] = ""
    return cache[cache_key]


def _read_excluded_metric_keys(metric_key_map: str | None, queries: pd.DataFrame) -> tuple[set[str], dict[str, str]]:
    target_keys = set(queries["molecule_id"].astype(str))
    if not metric_key_map:
        return set(), {}
    mapping = pd.read_parquet(metric_key_map, columns=["inchikey14", "metric_key"])
    raw_to_metric = dict(zip(mapping["inchikey14"].astype(str), mapping["metric_key"].astype(str)))
    excluded = {raw for raw, metric in raw_to_metric.items() if metric in target_keys}
    return excluded, raw_to_metric


def _query_groups(query_path: str, *, max_peaks: int) -> tuple[pd.DataFrame, list[dict]]:
    frame = pd.read_parquet(query_path, columns=QUERY_COLUMNS)
    required = set(QUERY_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Query table missing columns: {sorted(missing)}")

    # Make one spectrum representation per molecule/polarity. Taking the
    # maximum normalized peak evidence across acquisitions keeps peaks seen at
    # only one collision energy while avoiding over-counting repeated scans.
    query_records: dict[tuple[str, str], dict] = {}
    for row in frame.itertuples(index=False):
        molecule_id = str(row.molecule_id)
        mode = _text(row.ionization_mode).strip().lower()
        mass = neutral_mass(row.precursor_mz, row.adduct)
        if mass is None or not math.isfinite(mass) or mass <= 0:
            continue
        identity = (molecule_id, mode)
        state = query_records.setdefault(identity, {"masses": [], "peaks": defaultdict(float)})
        state["masses"].append(float(mass))
        vector = peak_vector(
            row.ms2_mzs,
            row.ms2_normalized_intensities,
            max_peaks=max_peaks,
            precursor_mz=float(row.precursor_mz),
        )
        for bin_id, intensity in vector.items():
            state["peaks"][bin_id] = max(state["peaks"][bin_id], intensity)

    groups = []
    for (molecule_id, mode), state in query_records.items():
        peaks = state["peaks"]
        norm = math.sqrt(sum(value * value for value in peaks.values()))
        vector = {key: value / norm for key, value in peaks.items()} if norm else {}
        groups.append({
            "molecule_id": molecule_id,
            "mode": mode,
            "mass": float(np.median(state["masses"])),
            "vector": vector,
        })
    if not groups:
        raise ValueError(f"No valid query spectra in {query_path}")
    return frame, groups


def _sparse_peak_matrix(vectors: list[dict[int, float]]) -> csr_matrix:
    row_ids, bin_ids, values = [], [], []
    for row_id, vector in enumerate(vectors):
        for bin_id, value in vector.items():
            if 0 <= bin_id < MASS_BINS:
                row_ids.append(row_id)
                bin_ids.append(bin_id)
                values.append(value)
    return csr_matrix(
        (np.asarray(values, dtype=np.float32),
         (np.asarray(row_ids, dtype=np.int32), np.asarray(bin_ids, dtype=np.int32))),
        shape=(len(vectors), MASS_BINS), dtype=np.float32,
    )


def _fingerprints(smiles_list: list[str], cache: dict[str, object]):
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, DataStructs

    rdBase.DisableLog("rdApp.warning")
    rdBase.DisableLog("rdApp.error")
    output = []
    for smiles in smiles_list:
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles)
            cache[smiles] = (
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FINGERPRINT_BITS)
                if mol is not None else None
            )
        output.append(cache[smiles])
    return output, DataStructs


def _trim_best(best: dict[str, tuple[float, str, float]], limit: int) -> None:
    if len(best) > limit * 2:
        kept = sorted(best.items(), key=lambda item: item[1][0], reverse=True)[:limit]
        best.clear()
        best.update(kept)


def retrieve_analogs(
    train_path: str,
    query_path: str,
    candidate_path: str,
    *,
    metric_key_map: str | None = None,
    output_features: str,
    output_predictions: str,
    max_delta_da: float = 200.0,
    ppm_window: float = 50.0,
    max_peaks: int = 256,
    batch_size: int = 4096,
    top_analogs: int = 128,
    top_spectra_per_batch: int = 256,
    similarity_power: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_frame, groups = _query_groups(query_path, max_peaks=max_peaks)
    excluded_raw_keys, raw_to_metric = _read_excluded_metric_keys(metric_key_map, query_frame)
    if metric_key_map:
        print(
            f"Excluding {len(excluded_raw_keys):,} raw aliases for "
            f"{query_frame['molecule_id'].nunique():,} held-out molecules",
            flush=True,
        )

    query_matrix = _sparse_peak_matrix([group["vector"] for group in groups])
    groups_by_mode: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        groups_by_mode[group["mode"]].append(index)

    candidates = pd.read_parquet(
        candidate_path, columns=["smiles", "metric_key", "exact_mass"]
    ).dropna(subset=["smiles", "metric_key", "exact_mass"])
    candidates["exact_mass"] = pd.to_numeric(candidates["exact_mass"], errors="coerce")
    candidates = candidates[np.isfinite(candidates["exact_mass"]) & (candidates["exact_mass"] > 0)]
    candidates = candidates.drop_duplicates("metric_key").sort_values("exact_mass").reset_index(drop=True)
    candidate_masses = candidates["exact_mass"].to_numpy(dtype=np.float64)

    # Keep only the strongest unique reference structures per query. The
    # dictionary is periodically pruned, so memory does not grow with the
    # number of reference spectra.
    best_by_molecule: dict[str, dict[str, tuple[float, str, float]]] = defaultdict(dict)
    parquet = pq.ParquetFile(train_path)
    scanned = 0
    next_progress = 250_000
    for batch in parquet.iter_batches(columns=TRAIN_COLUMNS, batch_size=batch_size):
        data = batch.to_pydict()
        count = batch.num_rows
        modes = np.asarray([_text(value).strip().lower() for value in data["ionization_mode"]], dtype=object)
        masses = np.fromiter(
            (neutral_mass(mz, adduct) or np.nan for mz, adduct in zip(data["precursor_mz"], data["adduct"])),
            dtype=np.float64,
            count=count,
        )
        raw_keys = np.asarray([_text(value) for value in data["inchikey14"]], dtype=object)
        allowed = np.isfinite(masses) & (raw_keys != "")
        if excluded_raw_keys:
            allowed &= np.fromiter((key not in excluded_raw_keys for key in raw_keys), dtype=bool, count=count)

        active = np.zeros(count, dtype=bool)
        for mode, indices in groups_by_mode.items():
            mode_mask = modes == mode
            for query_index in indices:
                active |= mode_mask & (np.abs(masses - groups[query_index]["mass"]) <= max_delta_da)
        active &= allowed
        active_rows = np.flatnonzero(active)

        if len(active_rows):
            vectors = [
                peak_vector(
                    data["ms2_mzs"][row],
                    data["ms2_normalized_intensities"][row],
                    max_peaks=max_peaks,
                    precursor_mz=data["precursor_mz"][row],
                )
                for row in active_rows
            ]
            reference_matrix = _sparse_peak_matrix(vectors)
            similarities = (reference_matrix @ query_matrix.T).toarray()

            for query_index, group in enumerate(groups):
                valid = (
                    (modes[active_rows] == group["mode"])
                    & (np.abs(masses[active_rows] - group["mass"]) <= max_delta_da)
                )
                positions = np.flatnonzero(valid)
                if not len(positions):
                    continue
                scores = similarities[positions, query_index]
                positive_positions = np.flatnonzero(scores > 0)
                if not len(positive_positions):
                    continue
                take = min(top_spectra_per_batch, len(positive_positions))
                chosen = positive_positions[np.argpartition(scores[positive_positions], -take)[-take:]]
                best = best_by_molecule[group["molecule_id"]]
                for selected in chosen:
                    position = int(positions[int(selected)])
                    row = int(active_rows[position])
                    score = float(scores[int(selected)])
                    raw_key = str(raw_keys[row])
                    # Use the raw key as a temporary identity without a map;
                    # the bounded analog pool is scorer-canonicalized later.
                    metric_key = raw_to_metric.get(raw_key) or raw_key
                    if not metric_key or metric_key == group["molecule_id"]:
                        continue
                    previous = best.get(metric_key)
                    if previous is None or score > previous[0]:
                        best[metric_key] = (
                            score,
                            str(data["normalized_smiles"][row]),
                            abs(float(masses[row]) - group["mass"]),
                        )
                _trim_best(best, top_analogs)

        scanned += count
        if scanned >= next_progress:
            print(
                f"Scanned {scanned:,}/{parquet.metadata.num_rows:,} reference spectra; "
                f"active in current batch={len(active_rows):,}",
                flush=True,
            )
            next_progress += 250_000

    fp_cache: dict[str, object] = {}
    key_cache: dict[str, str] = {}
    # Canonicalize only the bounded final analog pool. Running tautomer
    # enumeration for each batch winner would make the streamed scan costly.
    analogs_by_molecule = {}
    for molecule_id, analog_map in best_by_molecule.items():
        canonicalized: dict[str, tuple[float, str, float]] = {}
        for identity, evidence in sorted(analog_map.items(), key=lambda item: item[1][0], reverse=True):
            similarity, smiles, mass_delta = evidence
            metric_key = raw_to_metric.get(identity, "")
            if not metric_key:
                metric_key = _metric_key(smiles, identity, key_cache)
            if not metric_key or metric_key == molecule_id:
                continue
            previous = canonicalized.get(metric_key)
            if previous is None or similarity > previous[0]:
                canonicalized[metric_key] = (similarity, smiles, mass_delta)
            if len(canonicalized) >= top_analogs:
                break
        analogs_by_molecule[molecule_id] = sorted(
            canonicalized.items(), key=lambda item: item[1][0], reverse=True
        )[:top_analogs]
    feature_rows = []
    prediction_rows = []
    all_molecule_ids = list(dict.fromkeys(query_frame["molecule_id"].astype(str)))
    neutral_masses = defaultdict(list)
    for row in query_frame.itertuples(index=False):
        mass = neutral_mass(row.precursor_mz, row.adduct)
        if mass is not None and math.isfinite(mass) and mass > 0:
            neutral_masses[str(row.molecule_id)].append(float(mass))

    for molecule_id in all_molecule_ids:
        center = float(np.median(neutral_masses[molecule_id]))
        radius = max(center * ppm_window * 1e-6, 0.01)
        left = int(np.searchsorted(candidate_masses, center - radius, side="left"))
        right = int(np.searchsorted(candidate_masses, center + radius, side="right"))
        candidate_group = candidates.iloc[left:right]
        if candidate_group.empty:
            nearest = int(np.searchsorted(candidate_masses, center))
            candidate_group = candidates.iloc[max(0, nearest - 12):min(len(candidates), nearest + 13)]

        analogs = analogs_by_molecule.get(molecule_id, [])
        analog_smiles = [item[1][1] for item in analogs]
        analog_vectors, data_structs = _fingerprints(analog_smiles, fp_cache)
        analog_valid = [
            (analog_vectors[i], analogs[i][1][0], analogs[i][1][2])
            for i in range(len(analogs)) if analog_vectors[i] is not None
        ]
        scored = []
        candidate_smiles = candidate_group["smiles"].astype(str).tolist()
        candidate_vectors, data_structs = _fingerprints(candidate_smiles, fp_cache)
        for (_, candidate), candidate_fp in zip(candidate_group.iterrows(), candidate_vectors):
            key = str(candidate.metric_key)
            mass_error = abs(float(candidate.exact_mass) - center) / center * 1e6
            best_score = 0.0
            best_similarity = 0.0
            best_tanimoto = 0.0
            best_delta = np.nan
            if candidate_fp is not None and analog_valid:
                tanimotos = data_structs.BulkTanimotoSimilarity(
                    candidate_fp, [entry[0] for entry in analog_valid]
                )
                for (analog_fp, spectral_similarity, mass_delta), tanimoto in zip(analog_valid, tanimotos):
                    score = (spectral_similarity ** similarity_power) * float(tanimoto)
                    if score > best_score:
                        best_score = score
                        best_similarity = spectral_similarity
                        best_tanimoto = float(tanimoto)
                        best_delta = mass_delta
            feature_rows.append({
                "molecule_id": molecule_id,
                "metric_key": key,
                "smiles": str(candidate.smiles),
                "mass_error_ppm": mass_error,
                "analog_score": best_score,
                "analog_similarity": best_similarity,
                "analog_tanimoto": best_tanimoto,
                "analog_mass_delta_da": best_delta,
                "analog_reference_count": len(analog_valid),
            })
            scored.append((best_score, mass_error, key, str(candidate.smiles)))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        # Collapse aliases under the same metric key and produce a top-25 list.
        seen = set()
        smiles = []
        for _, _, key, structure in scored:
            metric_key = key or _metric_key(structure, "", key_cache)
            if not metric_key or metric_key in seen:
                continue
            seen.add(metric_key)
            smiles.append(structure)
            if len(smiles) == 25:
                break
        if not smiles:
            raise RuntimeError(f"No valid candidate structures for {molecule_id}")
        prediction_rows.append({"molecule_id": molecule_id, "smiles": ";".join(smiles)})

    feature_frame = pd.DataFrame(feature_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    feature_path, prediction_path = Path(output_features), Path(output_predictions)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_parquet(feature_path, index=False)
    prediction_frame.to_csv(prediction_path, index=False)
    print(
        f"Scanned {scanned:,} spectra; scored {len(feature_frame):,} candidates for "
        f"{len(prediction_frame):,} molecules. Wrote {feature_path} and {prediction_path}",
        flush=True,
    )
    return feature_frame, prediction_frame


def _metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float | int]:
    reciprocal_ranks = []
    hits = {1: 0, 5: 0, 25: 0}
    included = 0
    for _, group in frame.groupby("molecule_id", sort=False):
        ascending = score_column == "mass_error_ppm"
        ordered = group.sort_values(
            [score_column, "mass_error_ppm", "metric_key"],
            ascending=[ascending, True, True],
        )
        truth = np.flatnonzero(ordered["label"].to_numpy(dtype=bool))
        if not len(truth):
            continue
        included += 1
        rank = int(truth[0]) + 1
        reciprocal_ranks.append(1.0 / rank if rank <= 25 else 0.0)
        for k in hits:
            hits[k] += int(rank <= k)
    return {
        "molecules_with_truth_in_candidates": included,
        "mrr_at_25": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "hit_at_1": hits[1] / included if included else 0.0,
        "hit_at_5": hits[5] / included if included else 0.0,
        "hit_at_25": hits[25] / included if included else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--metric-key-map", help="Raw-to-scorer key map; excludes held-out target structures")
    parser.add_argument("--output-features", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--max-delta-da", type=float, default=200.0)
    parser.add_argument("--ppm-window", type=float, default=50.0)
    parser.add_argument("--max-peaks", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--top-analogs", type=int, default=128)
    parser.add_argument("--top-spectra-per-batch", type=int, default=256)
    parser.add_argument("--similarity-power", type=float, default=3.0)
    args = parser.parse_args()
    if min(args.max_delta_da, args.ppm_window, args.batch_size, args.top_analogs, args.top_spectra_per_batch) <= 0:
        parser.error("Mass windows, batch size, and analog caps must be positive")
    features, _ = retrieve_analogs(
        args.train, args.queries, args.candidates,
        metric_key_map=args.metric_key_map,
        output_features=args.output_features,
        output_predictions=args.output_predictions,
        max_delta_da=args.max_delta_da,
        ppm_window=args.ppm_window,
        max_peaks=args.max_peaks,
        batch_size=args.batch_size,
        top_analogs=args.top_analogs,
        top_spectra_per_batch=args.top_spectra_per_batch,
        similarity_power=args.similarity_power,
    )
    features["label"] = features["molecule_id"].astype(str) == features["metric_key"].astype(str)
    print("Analog-only ranking:", _metrics(features, "analog_score"))
    print("Mass-only ranking:", _metrics(features, "mass_error_ppm"))


if __name__ == "__main__":
    main()
