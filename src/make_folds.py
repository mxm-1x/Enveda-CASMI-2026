#!/usr/bin/env python3
"""Create deterministic structure-grouped folds from train.parquet."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import pyarrow.parquet as pq

from chemistry import metric_inchikey14


def fold_for_key(key: str, n_folds: int, seed: int) -> int:
    payload = f"{seed}:{key}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value % n_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", default="artifacts/folds.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    metric_by_source_key: dict[str, str] = {}
    parquet = pq.ParquetFile(args.train)
    for batch in parquet.iter_batches(columns=["inchikey14", "normalized_smiles"], batch_size=65_536):
        source_keys = batch.column(0).to_pylist()
        smiles_values = batch.column(1).to_pylist()
        for source_key, smiles in zip(source_keys, smiles_values):
            if source_key and smiles and source_key not in metric_by_source_key:
                metric_by_source_key[source_key] = metric_inchikey14(smiles)
    keys = {key for key in metric_by_source_key.values() if key}

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_inchikey14", "metric_inchikey14", "fold"])
        writer.writerows(
            (source_key, metric_key, fold_for_key(metric_key, args.folds, args.seed))
            for source_key, metric_key in sorted(metric_by_source_key.items())
        )
    print(f"Wrote {len(keys):,} metric-equivalent structure folds to {args.output}")


if __name__ == "__main__":
    main()
