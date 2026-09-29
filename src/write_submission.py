#!/usr/bin/env python3
"""Validate and write Kaggle's semicolon-delimited top-25 submission."""
from __future__ import annotations

import argparse

import pandas as pd

from chemistry import metric_inchikey14


def validate_predictions(pred: pd.DataFrame, test_ids) -> pd.DataFrame:
    if not {"molecule_id", "smiles"}.issubset(pred.columns):
        raise ValueError("Predictions must have molecule_id and smiles columns")
    if pred["molecule_id"].isna().any() or pred["smiles"].isna().any():
        raise ValueError("Submission columns cannot contain null values")
    if pred["molecule_id"].duplicated().any():
        raise ValueError("Each molecule_id must appear exactly once")
    expected, actual = set(test_ids), set(pred["molecule_id"])
    if expected != actual:
        raise ValueError(f"Molecule IDs differ from test: {len(expected-actual)} missing, {len(actual-expected)} extra")

    for molecule_id, value in zip(pred["molecule_id"], pred["smiles"]):
        candidates = [part.strip() for part in str(value).split(";") if part.strip()]
        if not candidates:
            raise ValueError(f"No candidate SMILES for {molecule_id}")
        if len(candidates) > 25:
            raise ValueError(f"{molecule_id} has {len(candidates)} candidates; maximum is 25")
        keys = [metric_inchikey14(candidate) for candidate in candidates]
        if any(not key for key in keys):
            raise ValueError(f"Invalid SMILES for {molecule_id}")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate connectivity candidates for {molecule_id}")
    return pred[["molecule_id", "smiles"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV with molecule_id and candidate SMILES list")
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    test = pd.read_parquet(args.test, columns=["molecule_id"])
    validate_predictions(pred, test["molecule_id"].unique())[["molecule_id", "smiles"]].to_csv(args.output, index=False)
    print(f"Wrote {len(pred):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
