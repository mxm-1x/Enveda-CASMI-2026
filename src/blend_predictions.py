#!/usr/bin/env python3
"""Fuse spectral retrieval and database reranking candidate lists."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chemistry import metric_inchikey14


def _candidates(value) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def blend_predictions(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
    test_ids,
    *,
    top_k: int = 25,
    diagnostics: pd.DataFrame | None = None,
    spectral_margin_threshold: float | None = None,
) -> tuple[pd.DataFrame, int]:
    """Put the database list first for ambiguous spectral retrievals.

    A spectral retriever normally emits 25 rows even when its top candidates
    have little separating evidence. Appending the database route in that case
    silently drops all database candidates. Optional per-molecule diagnostics
    let the caller choose the leading route from the spectral top-two margin.
    """
    if not 1 <= top_k <= 25:
        raise ValueError("top_k must be between 1 and 25")
    if spectral_margin_threshold is not None and diagnostics is None:
        raise ValueError("Spectral diagnostics are required when setting a margin threshold")
    for name, frame in (("primary", primary), ("fallback", fallback)):
        if not {"molecule_id", "smiles"}.issubset(frame.columns):
            raise ValueError(f"{name} predictions require molecule_id and smiles columns")
        if frame["molecule_id"].isna().any() or frame["molecule_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} predictions have null or duplicate molecule IDs")

    expected = [str(value) for value in test_ids]
    expected_set = set(expected)
    primary_by_id = dict(zip(primary["molecule_id"].astype(str), primary["smiles"]))
    fallback_by_id = dict(zip(fallback["molecule_id"].astype(str), fallback["smiles"]))
    for name, predictions in (("primary", primary_by_id), ("fallback", fallback_by_id)):
        actual = set(predictions)
        if actual != expected_set:
            raise ValueError(
                f"{name} molecule IDs differ from test: "
                f"{len(expected_set - actual)} missing, {len(actual - expected_set)} extra"
            )

    diagnostics_by_id = {}
    if diagnostics is not None:
        if not {"molecule_id", "spectral_margin"}.issubset(diagnostics.columns):
            raise ValueError("Diagnostics require molecule_id and spectral_margin columns")
        if diagnostics["molecule_id"].isna().any() or diagnostics["molecule_id"].astype(str).duplicated().any():
            raise ValueError("Diagnostics have null or duplicate molecule IDs")
        diagnostics_by_id = dict(zip(
            diagnostics["molecule_id"].astype(str),
            pd.to_numeric(diagnostics["spectral_margin"], errors="coerce").fillna(0.0),
        ))
        if set(diagnostics_by_id) != expected_set:
            raise ValueError("Diagnostics molecule IDs differ from test")

    from rdkit import rdBase

    rdBase.DisableLog("rdApp.warning")
    rdBase.DisableLog("rdApp.error")
    key_cache: dict[str, str] = {}

    def key(smiles: str) -> str:
        if smiles not in key_cache:
            key_cache[smiles] = metric_inchikey14(smiles)
        return key_cache[smiles]

    output = []
    filled = 0
    database_first = 0
    for molecule_id in expected:
        candidates: list[str] = []
        seen: set[str] = set()
        if spectral_margin_threshold is None:
            route_order = (primary_by_id[molecule_id], fallback_by_id[molecule_id])
        elif float(diagnostics_by_id[molecule_id]) < spectral_margin_threshold:
            route_order = (fallback_by_id[molecule_id], primary_by_id[molecule_id])
            database_first += 1
        else:
            route_order = (primary_by_id[molecule_id], fallback_by_id[molecule_id])
        for value in map(_candidates, route_order):
            for smiles in value:
                metric_key = key(smiles)
                if not metric_key:
                    raise ValueError(f"Invalid candidate SMILES for molecule {molecule_id}")
                if metric_key in seen:
                    continue
                if len(candidates) == top_k:
                    break
                candidates.append(smiles)
                seen.add(metric_key)
            if len(candidates) == top_k:
                break
        if not candidates:
            raise ValueError(f"No valid candidates for molecule {molecule_id}")
        primary_keys = {key(x) for x in _candidates(primary_by_id[molecule_id])}
        if any(key(smiles) not in primary_keys for smiles in candidates):
            filled += 1
        output.append({"molecule_id": molecule_id, "smiles": ";".join(candidates)})
    result = pd.DataFrame(output, columns=["molecule_id", "smiles"])
    result.attrs["database_first"] = database_first
    return result, filled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="High-confidence primary predictions")
    parser.add_argument("--fallback", required=True, help="Secondary ranked predictions")
    parser.add_argument("--test", required=True, help="Test parquet containing molecule_id")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--diagnostics", help="Per-molecule CSV from retrieve.py --diagnostics-output")
    parser.add_argument("--spectral-margin-threshold", type=float,
                        help="Put database candidates first when the spectral top-two margin is below this value")
    args = parser.parse_args()

    primary = pd.read_csv(args.primary)
    fallback = pd.read_csv(args.fallback)
    test_ids = pd.read_parquet(args.test, columns=["molecule_id"])["molecule_id"].unique()
    diagnostics = pd.read_csv(args.diagnostics) if args.diagnostics else None
    result, filled = blend_predictions(
        primary, fallback, test_ids, top_k=args.top_k,
        diagnostics=diagnostics,
        spectral_margin_threshold=args.spectral_margin_threshold,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(
        f"Wrote {len(result):,} molecules to {output}; database-first for "
        f"{result.attrs['database_first']:,}; other-route candidates included for {filled:,}"
    )


if __name__ == "__main__":
    main()
