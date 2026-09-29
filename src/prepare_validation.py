#!/usr/bin/env python3
"""Create streaming parquet splits for structure- or library-held-out retrieval.

The output query parquet has a `molecule_id` equal to the competition's
tautomer-canonicalized InChIKey14, allowing the normal inference path to score
all acquisitions for one held-out molecule together.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chemistry import metric_inchikey14

REF_COLUMNS = [
    "normalized_smiles", "inchikey14", "ionization_mode", "adduct",
    "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
]
QUERY_COLUMNS = [
    "molecule_id", "ionization_mode", "adduct", "precursor_mz",
    "ms2_mzs", "ms2_normalized_intensities",
]


def selected_for_holdout(key: str, fraction: float, seed: int) -> bool:
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little") / (2**64)
    return value < fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["structure", "source"], default="structure")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--query-source", default="enveda-np-examples")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        parser.error("--fraction must be between 0 and 1")

    parquet = pq.ParquetFile(args.train)
    source_keys = {}
    for batch in parquet.iter_batches(columns=["inchikey14", "normalized_smiles"], batch_size=args.batch_size):
        for raw_key, smiles in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if raw_key and smiles and raw_key not in source_keys:
                source_keys[raw_key] = metric_inchikey14(smiles)

    heldout_metric_keys = {
        metric_key for metric_key in set(source_keys.values())
        if metric_key and selected_for_holdout(metric_key, args.fraction, args.seed)
    }
    if args.mode == "source" and "ingest_lib" not in parquet.schema_arrow.names:
        raise ValueError("Source holdout requires the ingest_lib column")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_path, queries_path = out_dir / "reference.parquet", out_dir / "queries.parquet"
    ref_writer = query_writer = None
    ref_rows = query_rows = 0
    read_columns = list(dict.fromkeys(REF_COLUMNS + ["ingest_lib"]))
    for batch in parquet.iter_batches(columns=read_columns, batch_size=args.batch_size):
        data = batch.to_pydict()
        raw_keys = data["inchikey14"]
        metric_keys = [source_keys.get(key, "") for key in raw_keys]
        if args.mode == "structure":
            query_mask = [key in heldout_metric_keys for key in metric_keys]
        else:
            query_mask = [lib == args.query_source for lib in data["ingest_lib"]]

        query_idx = [i for i, selected in enumerate(query_mask) if selected]
        ref_idx = [i for i, selected in enumerate(query_mask) if not selected]
        if query_idx:
            q_data = {"molecule_id": [metric_keys[i] for i in query_idx]}
            q_data.update({name: [data[name][i] for i in query_idx] for name in QUERY_COLUMNS if name != "molecule_id"})
            q_table = pa.Table.from_pydict(q_data)
            if query_writer is None:
                query_writer = pq.ParquetWriter(queries_path, q_table.schema, compression="zstd")
            query_writer.write_table(q_table)
            query_rows += len(query_idx)
        if ref_idx:
            r_table = pa.Table.from_pydict({name: [data[name][i] for i in ref_idx] for name in REF_COLUMNS})
            if ref_writer is None:
                ref_writer = pq.ParquetWriter(refs_path, r_table.schema, compression="zstd")
            ref_writer.write_table(r_table)
            ref_rows += len(ref_idx)
    if ref_writer:
        ref_writer.close()
    if query_writer:
        query_writer.close()
    if not query_rows:
        raise RuntimeError("Validation split produced no queries")
    print(f"Mode={args.mode}; reference rows={ref_rows:,}; query rows={query_rows:,}; held-out structures={len(heldout_metric_keys):,}")
    print(f"Reference: {refs_path}\nQueries: {queries_path}")


if __name__ == "__main__":
    main()
