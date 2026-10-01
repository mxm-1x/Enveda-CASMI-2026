#!/usr/bin/env python3
"""Compare fragment-only and analog+fragment rankers on grouped holdouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from train_reranker import FEATURE_COLUMNS as FRAGMENT_FEATURES

ANALOG_FEATURES = [
    "analog_score",
    "analog_similarity",
    "analog_tanimoto",
    "analog_mass_delta_da",
    "analog_reference_count",
]
MULTICHANNEL_FEATURES = FRAGMENT_FEATURES + ANALOG_FEATURES


def _combine(base_path: str, analog_path: str) -> pd.DataFrame:
    base = pd.read_parquet(base_path)
    analog = pd.read_parquet(
        analog_path,
        columns=["molecule_id", "metric_key", *ANALOG_FEATURES],
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
    # No matching analog evidence is a real route outcome, represented by
    # neutral/zero evidence rather than dropping that candidate.
    result[ANALOG_FEATURES] = result[ANALOG_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    result["label"] = result["molecule_id"] == result["metric_key"]
    return result


def _fit(frame: pd.DataFrame, features: list[str], seed: int) -> HistGradientBoostingClassifier:
    x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int8)
    positive_weight = len(y) / max(2 * int(y.sum()), 1)
    negative_weight = len(y) / max(2 * int((y == 0).sum()), 1)
    weights = np.where(y == 1, positive_weight, negative_weight).astype(np.float32)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=seed,
    )
    return model.fit(x, y, sample_weight=weights)


def _metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float | int]:
    reciprocal_ranks = []
    hits = {1: 0, 5: 0, 25: 0}
    covered = 0
    for _, group in frame.groupby("molecule_id", sort=False):
        ordered = group.sort_values(
            [score_column, "mass_error_ppm", "metric_key"],
            ascending=[False, True, True],
        )
        truth = np.flatnonzero(ordered["label"].to_numpy(dtype=bool))
        if not len(truth):
            continue
        covered += 1
        rank = int(truth[0]) + 1
        reciprocal_ranks.append(1.0 / rank if rank <= 25 else 0.0)
        for k in hits:
            hits[k] += int(rank <= k)
    total = frame["molecule_id"].nunique()
    return {
        "molecules": int(total),
        "truth_in_candidate_pool": int(covered),
        "candidate_recall": covered / total if total else 0.0,
        "mrr_at_25": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "hit_at_1": hits[1] / covered if covered else 0.0,
        "hit_at_5": hits[5] / covered if covered else 0.0,
        "hit_at_25": hits[25] / covered if covered else 0.0,
    }


def _fold(molecule_id: str, n_folds: int) -> int:
    digest = hashlib.blake2b(molecule_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_folds


def evaluate(
    train_base: str,
    train_analog: str,
    valid_base: str,
    valid_analog: str,
    *,
    report_path: str,
    model_dir: str,
    seeds: list[int],
    folds: int = 5,
) -> dict:
    base_train = _combine(train_base, train_analog)
    base_valid = _combine(valid_base, valid_analog)
    eligible = base_train.groupby("molecule_id", sort=False)["label"].any()
    base_train = base_train[base_train["molecule_id"].isin(eligible[eligible].index)].copy()

    fold_ids = base_train["molecule_id"].map(lambda value: _fold(value, folds))
    oof_base = np.full(len(base_train), np.nan, dtype=np.float32)
    oof_multi = np.full(len(base_train), np.nan, dtype=np.float32)
    for fold_id in range(folds):
        valid_mask = fold_ids.to_numpy() == fold_id
        train_frame = base_train.loc[~valid_mask]
        valid_frame = base_train.loc[valid_mask]
        if not valid_frame["label"].any() or not train_frame["label"].any():
            raise RuntimeError(f"Fold {fold_id} lacks positive training or validation groups")
        for features, output in ((FRAGMENT_FEATURES, oof_base), (MULTICHANNEL_FEATURES, oof_multi)):
            model = _fit(train_frame, features, seed=20261001 + fold_id)
            matrix = valid_frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
            output[np.flatnonzero(valid_mask)] = model.predict_proba(matrix)[:, 1]

    base_train["oof_fragment_score"] = oof_base
    base_train["oof_multichannel_score"] = oof_multi
    report = {
        "training_molecules_with_truth": int(base_train["molecule_id"].nunique()),
        "training_rows": int(len(base_train)),
        "out_of_fold_fragment_ranker": _metrics(base_train, "oof_fragment_score"),
        "out_of_fold_multichannel_ranker": _metrics(base_train, "oof_multichannel_score"),
        "external_validation": {},
        "seeds": seeds,
        "features": MULTICHANNEL_FEATURES,
    }

    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    external_scores = {"fragment": [], "multichannel": []}
    x_valid_fragment = base_valid[FRAGMENT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    x_valid_multi = base_valid[MULTICHANNEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    for seed in seeds:
        model_base = _fit(base_train, FRAGMENT_FEATURES, seed)
        model_multi = _fit(base_train, MULTICHANNEL_FEATURES, seed)
        external_scores["fragment"].append(model_base.predict_proba(x_valid_fragment)[:, 1])
        external_scores["multichannel"].append(model_multi.predict_proba(x_valid_multi)[:, 1])
        import joblib
        joblib.dump(
            {"model": model_multi, "features": MULTICHANNEL_FEATURES, "seed": seed},
            model_root / f"analog_fragment_reranker_seed{seed}.joblib",
        )

    base_valid["fragment_model_score"] = np.mean(external_scores["fragment"], axis=0)
    base_valid["multichannel_model_score"] = np.mean(external_scores["multichannel"], axis=0)
    report["external_validation"] = {
        "fragment_ranker": _metrics(base_valid, "fragment_model_score"),
        "multichannel_ranker": _metrics(base_valid, "multichannel_model_score"),
        "analog_only": _metrics(base_valid, "analog_score"),
        "fragment_only": _metrics(base_valid, "fragment_score"),
    }

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    predictions = []
    for molecule_id, group in base_valid.groupby("molecule_id", sort=False):
        ordered = group.sort_values(
            ["multichannel_model_score", "mass_error_ppm", "metric_key"],
            ascending=[False, True, True],
        ).head(25)
        predictions.append({"molecule_id": molecule_id, "smiles": ";".join(ordered["smiles"].astype(str))})
    predictions_path = report_output.with_name("multichannel_external_predictions.csv")
    pd.DataFrame(predictions).to_csv(predictions_path, index=False)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Models: {model_root}\nPredictions: {predictions_path}\nReport: {report_output}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-base", required=True)
    parser.add_argument("--train-analog", required=True)
    parser.add_argument("--valid-base", required=True)
    parser.add_argument("--valid-analog", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261001, 20261002, 20261003])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    evaluate(
        args.train_base, args.train_analog, args.valid_base, args.valid_analog,
        report_path=args.report,
        model_dir=args.model_dir,
        seeds=args.seeds,
        folds=args.folds,
    )


if __name__ == "__main__":
    main()
