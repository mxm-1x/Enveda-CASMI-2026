#!/usr/bin/env python3
"""Fit the final CPU reranker on all prepared molecule-grouped training features."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import __version__ as sklearn_version

from train_reranker import FEATURE_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Prepared training candidate feature parquet")
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()

    frame = pd.read_parquet(args.features)
    required = {"molecule_id", "metric_key", *FEATURE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature table missing columns: {sorted(missing)}")
    frame["label"] = frame["molecule_id"].astype(str) == frame["metric_key"].astype(str)
    positives = frame.groupby("molecule_id", sort=False)["label"].any()
    eligible = set(positives[positives].index.astype(str))
    frame = frame[frame["molecule_id"].astype(str).isin(eligible)].copy()
    if frame.empty or not frame["label"].any() or frame["label"].all():
        raise RuntimeError("Training features must contain positive and negative candidates")

    x = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int8)
    positive_weight = len(y) / max(2 * int(y.sum()), 1)
    negative_weight = len(y) / max(2 * int((y == 0).sum()), 1)
    weights = np.where(y == 1, positive_weight, negative_weight).astype(np.float32)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=args.max_iter,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=args.seed,
    )
    model.fit(x, y, sample_weight=weights)

    output = Path(args.model_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": FEATURE_COLUMNS,
        "seed": args.seed,
        "training_molecules": int(frame["molecule_id"].nunique()),
        "training_rows": int(len(frame)),
        "sklearn_version": sklearn_version,
    }, output)
    print(
        f"Fit on {frame['molecule_id'].nunique():,} molecules and {len(frame):,} candidate rows; "
        f"saved {output} (scikit-learn {sklearn_version})"
    )


if __name__ == "__main__":
    main()
