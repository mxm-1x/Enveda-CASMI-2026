#!/usr/bin/env python3
"""Prepare source- or metric-structure-held-out retrieval validation queries."""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chemistry import metric_inchikey14

QUERY_COLUMNS = [
    "ionization_mode", "adduct", "precursor_mz",
    "ms2_mzs", "ms2_normalized_intensities",
]
QUERY_INPUT_COLUMNS = [
    "inchikey14", "normalized_smiles", *QUERY_COLUMNS,
]


def selected_for_holdout(key: str, fraction: float, seed: int) -> bool:
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little") / (2**64)
    return value < fraction


def _canonicalize_pair(item: tuple[str, str]) -> tuple[str, str]:
    raw_key, smiles = item
    # Warnings about enumerator caps and failed tautomer kekulization are
    # expected for some natural products; the returned metric key is still
    # computed with the competition-pinned RDKit version.
    from rdkit import rdBase
    rdBase.DisableLog("rdApp.warning")
    rdBase.DisableLog("rdApp.error")
    return raw_key, metric_inchikey14(smiles)


def _source_queries(args, out_dir: Path, queries_path: Path) -> None:
    table = pq.read_table(
        args.train,
        columns=QUERY_INPUT_COLUMNS,
        filters=[("ingest_lib", "=", args.query_source)],
    )
    data = table.to_pydict()
    key_cache: dict[str, str] = {}
    molecule_ids = []
    for raw_key, smiles in zip(data["inchikey14"], data["normalized_smiles"]):
        if raw_key not in key_cache:
            key_cache[raw_key] = metric_inchikey14(smiles) if raw_key and smiles else ""
        molecule_ids.append(key_cache[raw_key])
    keep = [i for i, key in enumerate(molecule_ids) if key]
    query_data = {"molecule_id": [molecule_ids[i] for i in keep]}
    query_data.update({name: [data[name][i] for i in keep] for name in QUERY_COLUMNS})
    if not keep:
        raise RuntimeError(f"No valid queries found for source {args.query_source!r}")
    pq.write_table(pa.Table.from_pydict(query_data), queries_path, compression="zstd")
    print(
        f"Mode=source; query rows={len(keep):,}; "
        f"held-out structures={len(set(query_data['molecule_id'])):,}"
    )
    print(f"Queries: {queries_path}\nReference: original training parquet (exclude source during retrieval)")


def _metric_key_map(args, parquet, out_dir: Path) -> dict[str, str]:
    map_path = out_dir / "metric_key_map.parquet"
    mapping: dict[str, str] = {}
    if map_path.exists():
        table = pq.read_table(map_path, columns=["inchikey14", "metric_key"])
        mapping = dict(zip(table["inchikey14"].to_pylist(), table["metric_key"].to_pylist()))
        print(f"Loaded cached metric-key map: {len(mapping):,} structures from {map_path}", flush=True)

    raw_to_smiles: dict[str, str] = {}
    scanned = 0
    for batch in parquet.iter_batches(columns=["inchikey14", "normalized_smiles"], batch_size=args.batch_size):
        for raw_key, smiles in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if raw_key and smiles:
                raw_to_smiles.setdefault(raw_key, smiles)
        scanned += batch.num_rows
        if scanned and scanned % 262_144 < batch.num_rows:
            print(f"Collected keys from {scanned:,}/{parquet.metadata.num_rows:,} rows", flush=True)

    items = [(key, smiles) for key, smiles in raw_to_smiles.items() if key not in mapping]
    total = len(items)
    print(
        f"Canonicalizing {total:,} uncached structures using {args.workers} CPU workers",
        flush=True,
    )

    def checkpoint() -> None:
        temporary_path = map_path.with_suffix(".partial.parquet")
        pq.write_table(
            pa.Table.from_pydict({"inchikey14": list(mapping), "metric_key": list(mapping.values())}),
            temporary_path,
            compression="zstd",
        )
        temporary_path.replace(map_path)

    if args.workers == 1:
        results = map(_canonicalize_pair, items)
        for done, (raw_key, metric_key) in enumerate(results, 1):
            mapping[raw_key] = metric_key
            if done % 5_000 == 0 or done == total:
                checkpoint()
                print(f"Canonicalized {done:,}/{total:,} structures", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(_canonicalize_pair, items, chunksize=64)
            for done, (raw_key, metric_key) in enumerate(results, 1):
                mapping[raw_key] = metric_key
                if done % 5_000 == 0 or done == total:
                    checkpoint()
                    print(f"Canonicalized {done:,}/{total:,} structures", flush=True)

    if total == 0 and not map_path.exists():
        checkpoint()
    print(f"Cached metric-key map: {map_path} ({len(mapping):,} structures)", flush=True)
    return mapping


def _structure_queries(args, parquet, out_dir: Path, queries_path: Path) -> None:
    metric_by_raw = _metric_key_map(args, parquet, out_dir)
    heldout_metric_keys = {
        metric_key for metric_key in set(metric_by_raw.values())
        if metric_key and selected_for_holdout(metric_key, args.fraction, args.seed)
    }
    heldout_raw_keys = {
        raw_key for raw_key, metric_key in metric_by_raw.items()
        if metric_key in heldout_metric_keys
    }
    if not heldout_raw_keys:
        raise RuntimeError("Structure holdout selected no source keys; increase --fraction")

    # Parquet's `in` filter selects held-out acquisitions directly. Save every
    # raw source key belonging to selected metric keys so retrieval can exclude
    # all tautomer variants while reading the original training parquet.
    table = pq.read_table(
        args.train,
        columns=QUERY_INPUT_COLUMNS,
        filters=[("inchikey14", "in", sorted(heldout_raw_keys))],
    )
    data = table.to_pydict()
    molecule_ids = [metric_by_raw.get(raw_key, "") for raw_key in data["inchikey14"]]
    keep = [i for i, key in enumerate(molecule_ids) if key]
    query_data = {"molecule_id": [molecule_ids[i] for i in keep]}
    query_data.update({name: [data[name][i] for i in keep] for name in QUERY_COLUMNS})
    if not keep:
        raise RuntimeError("Metric-key structure holdout produced no query spectra")
    pq.write_table(pa.Table.from_pydict(query_data), queries_path, compression="zstd")
    excluded_path = out_dir / "heldout_raw_keys.txt"
    excluded_path.write_text("\n".join(sorted(heldout_raw_keys)) + "\n", encoding="utf-8")
    print(
        f"Mode=structure; query rows={len(keep):,}; "
        f"held-out structures={len(heldout_metric_keys):,}; "
        f"excluded raw structure keys={len(heldout_raw_keys):,}"
    )
    print(f"Queries: {queries_path}\nExcluded keys: {excluded_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["structure", "source"], default="structure")
    parser.add_argument("--fraction", type=float, default=0.001)
    parser.add_argument("--query-source", default="enveda-np-examples")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        parser.error("--fraction must be between 0 and 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    parquet = pq.ParquetFile(args.train)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries_path = out_dir / "queries.parquet"
    if args.mode == "source":
        if "ingest_lib" not in parquet.schema_arrow.names:
            raise ValueError("Source holdout requires the ingest_lib column")
        _source_queries(args, out_dir, queries_path)
    else:
        _structure_queries(args, parquet, out_dir, queries_path)


if __name__ == "__main__":
    main()
