#!/usr/bin/env python3
"""Apply a saved tabular reranker to candidate-level feature rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = bundle["features"]
    frame = pd.read_parquet(args.features)
    required = {"molecule_id", "metric_key", "smiles", "mass_error_ppm", *feature_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature table missing columns: {sorted(missing)}")
    values = frame[feature_columns].replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
    frame["model_score"] = model.predict_proba(values.to_numpy(dtype="float32"))[:, 1]

    predictions = []
    for molecule_id, group in frame.groupby("molecule_id", sort=False):
        ordered = group.sort_values(
            ["model_score", "mass_error_ppm", "metric_key"],
            ascending=[False, True, True],
        ).head(25)
        predictions.append({"molecule_id": molecule_id, "smiles": ";".join(ordered["smiles"].astype(str))})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output, index=False)
    print(f"Wrote {len(predictions):,} molecule predictions to {output}")


if __name__ == "__main__":
    main()
