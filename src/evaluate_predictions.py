#!/usr/bin/env python3
"""Score predictions against a prepared validation queries parquet."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from chemistry import metric_inchikey14
from score import reciprocal_rank_at_25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="Validation parquet with molecule_id=truth metric key")
    parser.add_argument("--predictions", required=True, help="CSV containing molecule_id and semicolon-joined smiles")
    args = parser.parse_args()

    query = pd.read_parquet(args.queries, columns=["molecule_id"])
    truth_by_molecule = query.groupby("molecule_id", sort=False).size().index.to_list()
    pred = pd.read_csv(args.predictions)
    if not {"molecule_id", "smiles"}.issubset(pred.columns):
        raise ValueError("Prediction CSV requires molecule_id and smiles columns")
    pred_map = dict(zip(pred["molecule_id"], pred["smiles"]))
    ranks = []
    for molecule_id in truth_by_molecule:
        candidates = [part.strip() for part in str(pred_map.get(molecule_id, "")).split(";") if part.strip()]
        keys = []
        for smiles in candidates[:25]:
            key = metric_inchikey14(smiles)
            if key and key not in keys:
                keys.append(key)
        ranks.append(reciprocal_rank_at_25(keys, molecule_id))
    n = len(ranks)
    metrics = {
        "molecules": n,
        "mrr_at_25": sum(ranks) / n if n else 0.0,
        "hit_at_1": sum(rank == 1.0 for rank in ranks) / n if n else 0.0,
        "hit_at_5": sum(rank >= 0.2 for rank in ranks) / n if n else 0.0,
        "hit_at_25": sum(rank > 0 for rank in ranks) / n if n else 0.0,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
