#!/usr/bin/env python3
"""Convert an open natural-products CSV/ZIP into a compact scoring-keyed table."""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chemistry import candidate_structure

SMILES_COLUMNS = (
    "canonicalsmiles", "smiles", "isomericsmiles", "canonicalisomericsmiles",
    "smilescanonical", "structuresmiles", "canonicalsmilesnostereo",
)
ID_COLUMNS = ("coconutid", "compoundid", "identifier", "id")


def _normalized_column(name: str) -> str:
    return "".join(char for char in name.lower() if char.isalnum())


def _csv_chunks(path: Path, chunk_size: int):
    options = {
        "dtype": str,
        "keep_default_na": False,
        "chunksize": chunk_size,
        "encoding": "utf-8-sig",
    }
    if path.suffix.lower() != ".zip":
        yield from pd.read_csv(path, **options)
        return

    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"No CSV file found inside {path}")
        if len(members) > 1:
            preferred = [name for name in members if "coconut" in Path(name).name.lower()]
            if len(preferred) == 1:
                members = preferred
            else:
                raise ValueError(f"Expected one CSV in {path}; found {members}")
        with archive.open(members[0]) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig") as text:
            yield from pd.read_csv(text, **options)


def _resolve_column(columns, requested: str | None, candidates: tuple[str, ...], label: str) -> str:
    by_normalized = {_normalized_column(column): column for column in columns}
    if requested:
        if requested not in columns:
            raise ValueError(f"{label} column {requested!r} not found; available columns: {list(columns)}")
        return requested
    for candidate in candidates:
        if candidate in by_normalized:
            return by_normalized[candidate]
    raise ValueError(
        f"Could not identify the {label} column. Available columns: {list(columns)}. "
        f"Pass --{label.lower().replace(' ', '-')}-column explicitly."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="COCONUT CSV or CSV ZIP archive")
    parser.add_argument("--output", required=True, help="Output candidate parquet path")
    parser.add_argument("--smiles-column")
    parser.add_argument("--id-column")
    parser.add_argument("--source-version", default="unspecified", help="For example 2026-09")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    input_path, output_path = Path(args.input), Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
    if partial_path.exists():
        partial_path.unlink()

    chunks = iter(_csv_chunks(input_path, args.chunk_size))
    try:
        first = next(chunks)
    except StopIteration:
        raise ValueError("Input CSV is empty")
    smiles_column = _resolve_column(first.columns, args.smiles_column, SMILES_COLUMNS, "SMILES")
    id_column = None
    if args.id_column:
        id_column = _resolve_column(first.columns, args.id_column, (), "ID")
    else:
        normalized = {_normalized_column(column): column for column in first.columns}
        id_column = next((normalized[name] for name in ID_COLUMNS if name in normalized), None)

    writer = None
    seen_metric_keys: set[str] = set()
    input_rows = valid_rows = invalid_rows = 0
    try:
        for current in itertools.chain((first,), chunks):
            records = []
            for row in current.to_dict("records"):
                input_rows += 1
                smiles = str(row.get(smiles_column, "")).strip()
                if not smiles:
                    invalid_rows += 1
                    continue
                converted = candidate_structure(smiles)
                if converted is None:
                    invalid_rows += 1
                    continue
                metric_key, standardized, formula, exact_mass = converted
                if metric_key in seen_metric_keys:
                    continue
                seen_metric_keys.add(metric_key)
                valid_rows += 1
                candidate_id = str(row.get(id_column, "")).strip() if id_column else ""
                if not candidate_id:
                    candidate_id = f"row-{input_rows}"
                records.append({
                    "candidate_id": candidate_id,
                    "smiles": standardized,
                    "metric_key": metric_key,
                    "formula": formula,
                    "exact_mass": exact_mass,
                })
            if records:
                table = pa.Table.from_pylist(records)
                if writer is None:
                    writer = pq.ParquetWriter(partial_path, table.schema, compression="zstd")
                writer.write_table(table)
            print(
                f"Processed {input_rows:,} rows; kept {valid_rows:,} unique structures; "
                f"invalid/empty={invalid_rows:,}",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("No valid candidate structures were found")
    partial_path.replace(output_path)
    import rdkit
    manifest = {
        "source_file": input_path.name,
        "source_sha256": _sha256(input_path),
        "source_version": args.source_version,
        "license": "CC0",
        "rdkit_version": rdkit.__version__,
        "metric": "RDKit tautomer-canonicalized InChIKey14",
        "input_rows": input_rows,
        "unique_valid_structures": valid_rows,
        "invalid_or_empty_rows": invalid_rows,
        "output_file": output_path.name,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {valid_rows:,} candidate structures to {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
