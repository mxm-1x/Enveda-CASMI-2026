#!/usr/bin/env python3
"""Streaming precursor-aware spectral-library retrieval baseline.

This makes one bounded-memory pass over train.parquet and compares spectra
only at compatible mode and neutral precursor mass. It is a CPU baseline, not
an inference claim about the hidden natural-product distribution.
"""
from __future__ import annotations

import argparse
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from chemistry import metric_inchikey14
from spectra import cosine, neutral_loss_vector, peak_vector

# Monoisotopic adduct shifts relative to neutral molecular mass, in Da.
ADDUCT_SHIFT = {
    "[M+H]+": 1.007276, "[M+NH4]+": 18.033823, "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158, "[M-H]-": -1.007276, "[M+Cl]-": 34.969402,
    "[M+CH2O2-H]-": 44.998201, "[M-H2O+H]+": -17.003289,
    "[M-2H2O+H]+": -35.013854, "[M-H2O-H]-": -19.017841,
}


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def neutral_mass(precursor_mz, adduct):
    try:
        mz = float(precursor_mz)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(mz) or mz <= 0:
        return None
    shift = ADDUCT_SHIFT.get(_text(adduct))
    return mz - shift if shift is not None else None


def _load_queries(test_path: str, max_peaks: int):
    frame = pd.read_parquet(test_path)
    required = {"molecule_id", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct", "ionization_mode"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test parquet missing columns: {sorted(missing)}")
    queries = []
    for row in frame.to_dict("records"):
        precursor = row.get("precursor_mz")
        vector = peak_vector(row["ms2_mzs"], row["ms2_normalized_intensities"], max_peaks=max_peaks, precursor_mz=precursor)
        if not vector:
            warnings.warn(f"Empty spectrum for molecule {row['molecule_id']}")
        queries.append({
            "molecule_id": row["molecule_id"], "mode": _text(row.get("ionization_mode")),
            "adduct": _text(row.get("adduct")),
            "precursor": float(precursor) if pd.notna(precursor) else None,
            "neutral_mass": neutral_mass(precursor, row.get("adduct")),
            "vector": vector,
            "loss_vector": neutral_loss_vector(float(precursor), row["ms2_mzs"], row["ms2_normalized_intensities"], max_peaks=max_peaks) if pd.notna(precursor) else {},
        })
    return queries


def _make_index(queries):
    by_mode, by_raw = defaultdict(list), defaultdict(list)
    for query_id, query in enumerate(queries):
        if query["neutral_mass"] is not None:
            by_mode[query["mode"]].append((query["neutral_mass"], query_id))
        if query["precursor"] is not None and query["adduct"]:
            by_raw[(query["mode"], query["adduct"])].append((query["precursor"], query_id))

    def arrays(groups):
        result = {}
        for group, values in groups.items():
            values.sort()
            result[group] = (np.asarray([x[0] for x in values]), np.asarray([x[1] for x in values], dtype=np.int32))
        return result
    return arrays(by_mode), arrays(by_raw)


def _nearby(index, center, ppm):
    if index is None or center is None:
        return ()
    masses, query_ids = index
    width = center * ppm * 1e-6
    left = np.searchsorted(masses, center - width, side="left")
    right = np.searchsorted(masses, center + width, side="right")
    return query_ids[left:right]


def _metric_key(smiles, source_key, cache):
    cache_key = _text(source_key) or smiles
    if cache_key in cache:
        return cache[cache_key]
    try:
        key = metric_inchikey14(smiles)
    except RuntimeError:
        # Smoke-test fallback only. Kaggle must attach the pinned RDKit wheel.
        key = _text(source_key)
    cache[cache_key] = key
    return key


def retrieve(train_path, test_path, *, max_ppm=30.0, fallback_ppm=250.0, max_peaks=256, batch_size=8192, top_k=25, exclude_source=None):
    queries = _load_queries(test_path, max_peaks)
    mass_index, raw_index = _make_index(queries)
    per_query = [dict() for _ in queries]
    fallback = [dict() for _ in queries]
    key_cache = {}
    columns = ["normalized_smiles", "inchikey14", "ionization_mode", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
    if exclude_source:
        columns.append("ingest_lib")
    parquet = pq.ParquetFile(train_path)
    total = 0
    next_progress = 250_000

    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            if exclude_source and data["ingest_lib"][i] == exclude_source:
                total += 1
                continue
            smiles = data["normalized_smiles"][i]
            if not smiles:
                total += 1
                continue
            mode, adduct = _text(data["ionization_mode"][i]), _text(data["adduct"][i])
            precursor = data["precursor_mz"][i]
            ref_neutral = neutral_mass(precursor, adduct)
            candidate_queries = set()
            if ref_neutral is not None:
                candidate_queries.update(_nearby(mass_index.get(mode), ref_neutral, fallback_ppm))
            if precursor is not None and adduct:
                candidate_queries.update(_nearby(raw_index.get((mode, adduct)), float(precursor), fallback_ppm))
            if not candidate_queries:
                total += 1
                continue

            source_key = _text(data["inchikey14"][i])
            raw_key = source_key or smiles
            close_matches = []

            for query_id in candidate_queries:
                query_id = int(query_id)
                query = queries[query_id]
                if query["mode"] != mode:
                    continue
                qmass = query["neutral_mass"]
                rmass = ref_neutral
                if qmass is None or rmass is None:
                    if query["adduct"] != adduct or query["precursor"] is None or precursor is None:
                        continue
                    qmass, rmass = query["precursor"], float(precursor)
                ppm_error = abs(qmass - rmass) / max(qmass, 1e-9) * 1e6
                if ppm_error > fallback_ppm:
                    continue

                # Keep a cheap source-key fallback for loose mass matches.
                # Tautomer canonicalization is only needed for candidates
                # close enough to enter the ranked list.
                fallback_key = source_key or smiles
                old_fallback = fallback[query_id].get(fallback_key)
                if old_fallback is None or ppm_error < old_fallback[0]:
                    fallback[query_id][fallback_key] = (ppm_error, smiles)
                if ppm_error > max_ppm:
                    continue
                close_matches.append((query_id, ppm_error))

            # Spectra are expensive to decode and compare. Only build vectors
            # for rows inside the tight scoring window; loose mass-only
            # fallback rows were already recorded above.
            if not close_matches:
                total += 1
                continue
            ref_vector = peak_vector(data["ms2_mzs"][i], data["ms2_normalized_intensities"][i], max_peaks=max_peaks, precursor_mz=precursor)
            if not ref_vector:
                total += 1
                continue
            ref_loss = neutral_loss_vector(float(precursor), data["ms2_mzs"][i], data["ms2_normalized_intensities"][i], max_peaks=max_peaks) if precursor is not None else {}

            for query_id, ppm_error in close_matches:
                query = queries[query_id]
                adduct = _text(data["adduct"][i])
                mass_weight = math.exp(-0.5 * (ppm_error / max(max_ppm, 1e-6)) ** 2)
                score = mass_weight * (0.78 * cosine(query["vector"], ref_vector) + 0.22 * cosine(query["loss_vector"], ref_loss))
                if query["adduct"] == adduct:
                    score *= 1.03
                old = per_query[query_id].get(raw_key)
                if score > 0 and (old is None or score > old[0]):
                    per_query[query_id][raw_key] = (score, smiles)
            total += 1
        if total >= next_progress:
            print(f"Scanned {total:,}/{parquet.metadata.num_rows:,} training rows", flush=True)
            next_progress += 250_000

    grouped_queries = defaultdict(list)
    for query_id, query in enumerate(queries):
        grouped_queries[query["molecule_id"]].append(query_id)

    rows = []
    for molecule_id, query_ids in grouped_queries.items():
        raw_evidence, smiles_by_raw_key = defaultdict(dict), {}
        for query_id in query_ids:
            for raw_key, (score, smiles) in per_query[query_id].items():
                raw_evidence[raw_key][query_id] = max(score, raw_evidence[raw_key].get(query_id, 0.0))
                smiles_by_raw_key.setdefault(raw_key, smiles)
        if raw_evidence:
            raw_ranked = []
            for raw_key, by_query in raw_evidence.items():
                scores = sorted(by_query.values(), reverse=True)
                top = scores[:min(3, len(query_ids))]
                combined = 0.7 * sum(top) / len(top) + 0.3 * scores[0]
                raw_ranked.append((combined, raw_key, smiles_by_raw_key[raw_key], by_query))
            raw_ranked.sort(reverse=True)
            # Canonicalize only a bounded high-scoring pool, rather than every
            # mass-compatible training structure during the full parquet scan.
            metric_evidence, smiles_by_metric_key = defaultdict(dict), {}
            pool_size = top_k
            for _, raw_key, smiles, by_query in raw_ranked[:pool_size]:
                metric_key = _metric_key(smiles, raw_key, key_cache)
                if not metric_key:
                    continue
                for query_id, score in by_query.items():
                    metric_evidence[metric_key][query_id] = max(
                        score, metric_evidence[metric_key].get(query_id, 0.0)
                    )
                smiles_by_metric_key.setdefault(metric_key, smiles)
            ranked = []
            for metric_key, by_query in metric_evidence.items():
                scores = sorted(by_query.values(), reverse=True)
                top = scores[:min(3, len(query_ids))]
                combined = 0.7 * sum(top) / len(top) + 0.3 * scores[0]
                ranked.append((combined, metric_key, smiles_by_metric_key[metric_key]))
            ranked.sort(reverse=True)
            candidates = [smiles for _, _, smiles in ranked[:top_k]]
        else:
            closest = {}
            for query_id in query_ids:
                for key, (ppm_error, smiles) in fallback[query_id].items():
                    if key not in closest or ppm_error < closest[key][0]:
                        closest[key] = (ppm_error, smiles)
            fallback_ranked = sorted(
                ((ppm_error, raw_key, smiles) for raw_key, (ppm_error, smiles) in closest.items())
            )
            seen_metric_keys = set()
            candidates = []
            for _, raw_key, smiles in fallback_ranked[:top_k]:
                metric_key = _metric_key(smiles, raw_key, key_cache)
                if metric_key and metric_key not in seen_metric_keys:
                    seen_metric_keys.add(metric_key)
                    candidates.append(smiles)
                    if len(candidates) == top_k:
                        break
        if not candidates:
            raise RuntimeError(f"No candidates for {molecule_id}; increase fallback_ppm or add a structure candidate database")
        rows.append({"molecule_id": molecule_id, "smiles": ";".join(candidates)})

    print(f"Scanned {total:,} training rows; canonicalized {len(key_cache):,} candidate structures")
    return pd.DataFrame(rows, columns=["molecule_id", "smiles"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--max-ppm", type=float, default=30.0)
    parser.add_argument("--fallback-ppm", type=float, default=250.0)
    parser.add_argument("--max-peaks", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--exclude-source", help="Skip this ingest_lib while retrieving, for source-held-out validation")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 25:
        parser.error("--top-k must be between 1 and 25")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result = retrieve(args.train, args.test, max_ppm=args.max_ppm, fallback_ppm=args.fallback_ppm, max_peaks=args.max_peaks, batch_size=args.batch_size, top_k=args.top_k, exclude_source=args.exclude_source)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} molecule predictions to {args.output}")


if __name__ == "__main__":
    main()
