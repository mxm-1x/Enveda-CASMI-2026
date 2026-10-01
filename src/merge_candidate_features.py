#!/usr/bin/env python3
"""Join independently generated candidate evidence tables by molecule/key."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANALOG_COLUMNS = [
    "analog_score",
    "analog_similarity",
    "analog_tanimoto",
    "analog_mass_delta_da",
    "analog_reference_count",
]


def merge_candidate_features(base_path: str, analog_path: str, output_path: str) -> pd.DataFrame:
    base = pd.read_parquet(base_path)
    analog = pd.read_parquet(
        analog_path,
        columns=["molecule_id", "metric_key", *ANALOG_COLUMNS],
    )
    for frame in (base, analog):
        frame["molecule_id"] = frame["molecule_id"].astype(str)
        frame["metric_key"] = frame["metric_key"].astype(str)
    result = base.merge(
        analog,
        on=["molecule_id", "metric_key"],
        how="left",
        validate="one_to_one",
    )
    matched = int(result["analog_score"].notna().sum())
    result[ANALOG_COLUMNS] = result[ANALOG_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    truth_groups = result.assign(
        label=result["molecule_id"] == result["metric_key"]
    ).groupby("molecule_id", sort=False)["label"].any().sum()
    print(
        f"Merged {matched:,}/{len(base):,} base candidates with analog evidence; "
        f"{result['molecule_id'].nunique():,} molecules, {truth_groups:,} with a true candidate. "
        f"Wrote {output}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Fragment/structure candidate features")
    parser.add_argument("--analog", required=True, help="Mass-shifted analog candidate features")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_candidate_features(args.base, args.analog, args.output)


if __name__ == "__main__":
    main()
