#!/usr/bin/env python3
"""Convert an open natural-products CSV/ZIP into a compact scoring-keyed table."""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import bisect
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chemistry import candidate_structure
from retrieve import neutral_mass

SMILES_COLUMNS = (
    "canonicalsmiles", "smiles", "isomericsmiles", "canonicalisomericsmiles",
    "smilescanonical", "structuresmiles", "canonicalsmilesnostereo",
)
ID_COLUMNS = ("coconutid", "compoundid", "identifier", "id")


def _convert_candidate(task):
    row_number, candidate_id, smiles = task
    return row_number, candidate_id, candidate_structure(smiles)


def _query_mass_centers(path: Path) -> list[float]:
    frame = pd.read_parquet(path, columns=["molecule_id", "precursor_mz", "adduct"])
    masses_by_molecule: dict[str, list[float]] = {}
    for molecule_id, precursor, adduct in frame.itertuples(index=False, name=None):
        mass = neutral_mass(precursor, adduct)
        if mass is not None and np.isfinite(mass) and mass > 0:
            masses_by_molecule.setdefault(str(molecule_id), []).append(float(mass))
    return sorted(float(np.median(masses)) for masses in masses_by_molecule.values() if masses)


def _within_query_mass_window(candidate_masses: np.ndarray, centers: list[float], window_ppm: float) -> np.ndarray:
    """Fast filter using a sorted set of neutral precursor mass centers."""
    result = np.zeros(len(candidate_masses), dtype=bool)
    for i, mass in enumerate(candidate_masses):
        if not np.isfinite(mass) or mass <= 0:
            continue
        position = bisect.bisect_left(centers, float(mass))
        for j in (position - 1, position):
            if 0 <= j < len(centers):
                center = centers[j]
                if abs(float(mass) - center) / center * 1e6 <= window_ppm:
                    result[i] = True
                    break
    return result


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
    parser.add_argument("--workers", type=int, default=4, help="Parallel RDKit workers; use 1 to disable")
    parser.add_argument("--chunksize", type=int, default=16, help="Structures submitted per worker task")
    parser.add_argument(
        "--query-parquet",
        help="Optional query spectra parquet; prefilter by exact_molecular_weight before RDKit",
    )
    parser.add_argument("--window-ppm", type=float, default=50.0)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.workers < 1 or args.chunksize < 1:
        parser.error("--workers and --chunksize must be positive")
    if args.window_ppm <= 0:
        parser.error("--window-ppm must be positive")

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
    exact_mass_column = None
    mass_centers = None
    if args.query_parquet:
        mass_columns = {_normalized_column(column): column for column in first.columns}
        exact_mass_column = mass_columns.get("exactmolecularweight")
        if not exact_mass_column:
            raise ValueError("Mass-window filtering requires an exact_molecular_weight input column")
        mass_centers = _query_mass_centers(Path(args.query_parquet))
        if not mass_centers:
            raise ValueError("No valid neutral query precursor masses found in --query-parquet")
        print(
            f"Prefiltering candidates against {len(mass_centers):,} query mass centers "
            f"within {args.window_ppm:g} ppm",
            flush=True,
        )
    id_column = None
    if args.id_column:
        id_column = _resolve_column(first.columns, args.id_column, (), "ID")
    else:
        normalized = {_normalized_column(column): column for column in first.columns}
        id_column = next((normalized[name] for name in ID_COLUMNS if name in normalized), None)

    writer = None
    seen_metric_keys: set[str] = set()
    input_rows = candidate_rows = valid_rows = invalid_rows = mass_filtered_rows = 0
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for current in itertools.chain((first,), chunks):
            records = []
            tasks = []
            input_rows += len(current)
            if mass_centers is not None and exact_mass_column:
                input_masses = pd.to_numeric(current[exact_mass_column], errors="coerce").to_numpy(dtype=np.float64)
                mask = _within_query_mass_window(input_masses, mass_centers, args.window_ppm)
                mass_filtered_rows += int((~mask).sum())
                current = current.loc[mask]
            for row in current.to_dict("records"):
                candidate_rows += 1
                smiles = str(row.get(smiles_column, "")).strip()
                if not smiles:
                    invalid_rows += 1
                    continue
                candidate_id = str(row.get(id_column, "")).strip() if id_column else ""
                if not candidate_id:
                    candidate_id = f"row-{candidate_rows}"
                tasks.append((candidate_rows, candidate_id, smiles))

            converted_tasks = (
                pool.map(_convert_candidate, tasks, chunksize=args.chunksize)
                if pool is not None
                else map(_convert_candidate, tasks)
            )
            for row_number, candidate_id, converted in converted_tasks:
                if converted is None:
                    invalid_rows += 1
                    continue
                metric_key, standardized, formula, exact_mass = converted
                if metric_key in seen_metric_keys:
                    continue
                seen_metric_keys.add(metric_key)
                valid_rows += 1
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
                f"Scanned {input_rows:,} rows; mass-filtered {mass_filtered_rows:,}; "
                f"kept {valid_rows:,} unique structures; "
                f"invalid/empty={invalid_rows:,}",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
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
        "mass_filter_query": str(Path(args.query_parquet).name) if args.query_parquet else None,
        "mass_filter_window_ppm": args.window_ppm if args.query_parquet else None,
        "mass_filtered_rows": mass_filtered_rows,
        "output_file": output_path.name,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {valid_rows:,} candidate structures to {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
