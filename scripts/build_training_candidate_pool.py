#!/usr/bin/env python3
"""Build an internal candidate pool from unique training structures.

This is for closed-world, structure-held-out development validation only. Use
prepare_candidates.py for an independent external database such as COCONUT.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem


FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def exact_mass_from_formula(formula: str, periodic_table) -> float | None:
    tokens = FORMULA_TOKEN.findall(formula)
    if not tokens or "".join(element + count for element, count in tokens) != formula:
        return None
    mass = 0.0
    for element, count in tokens:
        atomic_number = periodic_table.GetAtomicNumber(element)
        if not atomic_number:
            return None
        mass += periodic_table.GetMostCommonIsotopeMass(atomic_number) * int(count or 1)
    return mass


def build_candidate_pool(train_path: Path, map_path: Path, output_path: Path, batch_size: int) -> tuple[int, int]:
    key_table = pq.read_table(map_path, columns=["inchikey14", "metric_key"])
    metric_by_raw = dict(zip(key_table["inchikey14"].to_pylist(), key_table["metric_key"].to_pylist()))

    raw_structures: dict[str, tuple[str, str]] = {}
    parquet = pq.ParquetFile(train_path)
    for batch in parquet.iter_batches(
        columns=["inchikey14", "normalized_smiles", "molecular_formula"],
        batch_size=batch_size,
    ):
        for raw_key, smiles, formula in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist(), batch.column(2).to_pylist()
        ):
            if raw_key and smiles and formula:
                raw_structures.setdefault(raw_key, (smiles, formula))

    periodic_table = Chem.GetPeriodicTable()
    by_metric_key: dict[str, tuple[str, str, float]] = {}
    rejected_formulas = 0
    for raw_key, (smiles, formula) in raw_structures.items():
        metric_key = metric_by_raw.get(raw_key, "")
        if not metric_key or metric_key in by_metric_key:
            continue
        exact_mass = exact_mass_from_formula(formula, periodic_table)
        if exact_mass is None:
            rejected_formulas += 1
            continue
        by_metric_key[metric_key] = (smiles, formula, exact_mass)

    columns = {"metric_key": [], "smiles": [], "formula": [], "exact_mass": []}
    for metric_key, (smiles, formula, exact_mass) in by_metric_key.items():
        columns["metric_key"].append(metric_key)
        columns["smiles"].append(smiles)
        columns["formula"].append(formula)
        columns["exact_mass"].append(exact_mass)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
    pq.write_table(pa.Table.from_pydict(columns), partial_path, compression="zstd")
    partial_path.replace(output_path)
    print(
        f"Training raw structures={len(raw_structures):,}; candidate structures={len(by_metric_key):,}; "
        f"formulas rejected={rejected_formulas:,}; wrote {output_path}"
    )
    return len(by_metric_key), rejected_formulas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--metric-key-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    build_candidate_pool(Path(args.train), Path(args.metric_key_map), Path(args.output), args.batch_size)


if __name__ == "__main__":
    main()
