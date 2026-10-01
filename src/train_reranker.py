#!/usr/bin/env python3
"""Train a small molecule-grouped gradient-boosted candidate reranker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURE_COLUMNS = [
    "mass_error_ppm",
    "fragment_score",
    "fragment_intensity_sum",
    "matched_fragment_count",
    "predicted_fragment_count",
    "heavy_atom_count",
    "hetero_atom_count",
    "ring_count",
    "aromatic_ring_count",
    "rotatable_bond_count",
]


def validation_group(molecule_id: str, seed: int, folds: int = 5) -> bool:
    digest = hashlib.blake2b(f"{seed}:{molecule_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % folds == 0


def ranking_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float | int]:
    reciprocal_ranks = []
    hits = {1: 0, 5: 0, 25: 0}
    count = 0
    for _, group in frame.groupby("molecule_id", sort=False):
        ordered = group.sort_values([score_column, "mass_error_ppm", "metric_key"], ascending=[False, True, True])
        truth = np.flatnonzero(ordered["label"].to_numpy(dtype=bool))
        if not len(truth):
            continue
        count += 1
        rank = int(truth[0]) + 1
        reciprocal_ranks.append(1.0 / rank if rank <= 25 else 0.0)
        for k in hits:
            hits[k] += int(rank <= k)
    return {
        "molecules": count,
        "mrr_at_25": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "hit_at_1": hits[1] / count if count else 0.0,
        "hit_at_5": hits[5] / count if count else 0.0,
        "hit_at_25": hits[25] / count if count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Parquet from retrieve_fragments.py --features-output")
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--max-iter", type=int, default=120)
    args = parser.parse_args()

    frame = pd.read_parquet(args.features)
    required = {"molecule_id", "metric_key", *FEATURE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature table missing columns: {sorted(missing)}")
    frame["label"] = frame["molecule_id"].astype(str) == frame["metric_key"].astype(str)
    positive_by_group = frame.groupby("molecule_id", sort=False)["label"].any()
    eligible_ids = set(positive_by_group[positive_by_group].index.astype(str))
    frame = frame[frame["molecule_id"].astype(str).isin(eligible_ids)].copy()
    frame["molecule_id"] = frame["molecule_id"].astype(str)
    frame["metric_key"] = frame["metric_key"].astype(str)
    frame["is_validation"] = frame["molecule_id"].map(lambda value: validation_group(value, args.seed))
    training = frame[~frame["is_validation"]]
    validation = frame[frame["is_validation"]].copy()
    if training.empty or validation.empty or not training["label"].any() or not validation["label"].any():
        raise RuntimeError("Group split must contain positive candidate examples in both train and validation")

    x_train = training[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y_train = training["label"].to_numpy(dtype=np.int8)
    positive_weight = len(y_train) / max(2 * int(y_train.sum()), 1)
    negative_weight = len(y_train) / max(2 * int((y_train == 0).sum()), 1)
    weights = np.where(y_train == 1, positive_weight, negative_weight).astype(np.float32)

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=args.max_iter,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=args.seed,
    )
    model.fit(x_train, y_train, sample_weight=weights)
    x_validation = validation[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    validation["model_score"] = model.predict_proba(x_validation)[:, 1]

    metrics = {
        "fragment_baseline": ranking_metrics(validation, "fragment_score"),
        "hist_gradient_boosting": ranking_metrics(validation, "model_score"),
        "training_molecules": int(training["molecule_id"].nunique()),
        "validation_molecules_with_truth_candidate": int(validation["molecule_id"].nunique()),
        "all_molecules_with_truth_candidate": len(eligible_ids),
        "molecules_missing_truth_in_candidate_window": int(positive_by_group.size - len(eligible_ids)),
        "training_candidate_rows": int(len(training)),
        "validation_candidate_rows": int(len(validation)),
        "features": FEATURE_COLUMNS,
        "seed": args.seed,
    }

    predictions = []
    all_x = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    frame["model_score"] = model.predict_proba(all_x)[:, 1]
    for molecule_id, group in frame.groupby("molecule_id", sort=False):
        ordered = group.sort_values(["model_score", "mass_error_ppm", "metric_key"], ascending=[False, True, True]).head(25)
        predictions.append({"molecule_id": molecule_id, "smiles": ";".join(ordered["smiles"].tolist()) if "smiles" in ordered else ""})

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "seed": args.seed}, model_path)
    prediction_path = Path(args.predictions_output)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(prediction_path, index=False)
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Model: {model_path}\nPredictions: {prediction_path}\nReport: {report_path}")


if __name__ == "__main__":
    main()
