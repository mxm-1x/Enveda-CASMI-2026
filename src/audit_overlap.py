#!/usr/bin/env python3
"""Audit exact spectrum overlap for the downloadable dummy test only."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def signature(mzs, intensities):
    if len(mzs) != len(intensities):
        return None
    if not mzs:
        return (0,)
    return (len(mzs), mzs[0], mzs[-1], intensities[0], intensities[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="reports/dummy_overlap.json")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    test = pq.read_table(args.test, columns=["molecule_id", "ms2_mzs", "ms2_normalized_intensities"]).to_pydict()
    signature_to_test_rows = defaultdict(list)
    for row_id, (mzs, intensities) in enumerate(zip(test["ms2_mzs"], test["ms2_normalized_intensities"])):
        signature_to_test_rows[signature(mzs, intensities)].append(row_id)

    matched = defaultdict(list)
    source_counts = Counter()
    possible_hits = 0
    columns = ["ingest_lib", "inchikey14", "normalized_smiles", "ms2_mzs", "ms2_normalized_intensities"]
    parquet = pq.ParquetFile(args.train)
    for batch in parquet.iter_batches(columns=columns, batch_size=args.batch_size):
        data = batch.to_pydict()
        source_counts.update(lib for lib in data["ingest_lib"] if lib)
        for lib, key, smiles, mzs, intensities in zip(*(data[name] for name in columns)):
            sig = signature(mzs, intensities)
            row_ids = signature_to_test_rows.get(sig, ())
            if row_ids:
                possible_hits += len(row_ids)
                # The short signature only narrows candidates. Full aligned
                # arrays must match before an overlap is counted.
                for row_id in row_ids:
                    if mzs == test["ms2_mzs"][row_id] and intensities == test["ms2_normalized_intensities"][row_id]:
                        matched[row_id].append({"ingest_lib": lib, "inchikey14": key, "smiles": smiles})

    molecule_rows = defaultdict(list)
    for row_id, molecule_id in enumerate(test["molecule_id"]):
        molecule_rows[molecule_id].append(row_id)
    conflicting = []
    for molecule_id, row_ids in molecule_rows.items():
        labels = {
            match["inchikey14"]
            for row_id in row_ids
            for match in matched.get(row_id, [])
            if match["inchikey14"]
        }
        if len(labels) > 1:
            conflicting.append(molecule_id)

    report = {
        "warning": "Downloadable test.parquet is a dummy; these are plumbing checks, not leaderboard validation.",
        "test_spectra": len(test["molecule_id"]),
        "test_molecules": len(molecule_rows),
        "exactly_matched_spectra": len(matched),
        "molecules_with_any_match": sum(any(row in matched for row in rows) for rows in molecule_rows.values()),
        "matching_train_source_counts": dict(Counter(match["ingest_lib"] for rows in matched.values() for match in rows)),
        "conflicting_label_molecules": conflicting,
        "full_array_comparisons_after_signature_filter": possible_hits,
        "training_source_counts": dict(source_counts),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
