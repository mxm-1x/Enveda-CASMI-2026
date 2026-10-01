#!/usr/bin/env python3
"""Preserve primary rankings and use a secondary route only to fill open slots."""
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
) -> tuple[pd.DataFrame, int]:
    """Keep the primary order, then append unique fallback structures to top_k."""
    if not 1 <= top_k <= 25:
        raise ValueError("top_k must be between 1 and 25")
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
    for molecule_id in expected:
        candidates: list[str] = []
        seen: set[str] = set()
        for value in (_candidates(primary_by_id[molecule_id]), _candidates(fallback_by_id[molecule_id])):
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
        if len(candidates) > len(_candidates(primary_by_id[molecule_id])):
            filled += 1
        output.append({"molecule_id": molecule_id, "smiles": ";".join(candidates)})
    return pd.DataFrame(output, columns=["molecule_id", "smiles"]), filled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="High-confidence primary predictions")
    parser.add_argument("--fallback", required=True, help="Secondary ranked predictions")
    parser.add_argument("--test", required=True, help="Test parquet containing molecule_id")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    primary = pd.read_csv(args.primary)
    fallback = pd.read_csv(args.fallback)
    test_ids = pd.read_parquet(args.test, columns=["molecule_id"])["molecule_id"].unique()
    result, filled = blend_predictions(primary, fallback, test_ids, top_k=args.top_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result):,} molecules to {output}; fallback filled open slots for {filled:,}")


if __name__ == "__main__":
    main()
