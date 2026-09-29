# Enveda CASMI 2026: base plan and competitive strategy

## What the available files tell us

The local folder `enveda-CASMI26-molecule-id-mass-spectra/` contains 2,539,608 labeled training spectra (2.8 GB), 1,213 visible test spectra, and a 400-molecule sample submission. The visible test has 1–9 acquisitions per molecule (median 3), all on timsTOF. A complete streamed scan found that all 1,213 visible test spectra exactly match one `enveda-180` training row each; the matched labels agree within each molecule.

**The visible test file is a dummy.** The [competition data page](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) says it is drawn from training and replaced by a hidden set during notebook rerun. A [host clarification](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741404) confirms the leaderboard uses the hidden set. The local overlap result checks data plumbing only. It must not guide model selection or expected-score claims.

The hidden set targets natural-product-like chemistry and mixes three novelty classes: structures with public reference spectra, known structures without public reference spectra, and structures absent from PubChem. Their proportions are hidden. Scoring is molecule-level MRR@25 after RDKit 2026.03.3 tautomer canonicalization and InChIKey14 comparison.

## Winning hypothesis

Build one candidate-ranking system with three evidence channels:

1. **Spectral library:** indexed search over 2.54M training spectra, with precursor/adduct gating, tolerant fragment and neutral-loss similarity, and aggregation across acquisitions. This covers novelty class 1.
2. **Structure database:** formula or narrow exact-mass candidates from public PubChem and COCONUT snapshots, standardized to competition connectivity keys. Predict molecular fingerprints from spectra, then rank candidates using structure-aware fragment evidence. This covers much of class 2.
3. **Novelty:** reserve a small, calibrated number of positions for generated or analog-derived candidates only when validation on genuinely absent structures shows gains. Exact new connectivity is difficult; this starts as research rather than the baseline.

A final calibrated ranker combines evidence and deduplicates by the same tautomer-normalized InChIKey14 used by the metric. Track candidate recall separately from ranking quality. If the truth is absent from the candidate pool, ranking cannot recover it.

## Validation that predicts the hidden set

| Slice | Construction | Question |
|---|---|---|
| Library-known natural products | Natural-product spectra from `enveda-np-examples`, RIKEN, GNPS, and MassBank as queries; exclude identical acquisitions but retain independent reference spectra for the compound | Can library search recover a known natural product across instruments? |
| Database-known, spectra absent | Hold out every spectrum for selected InChIKey14 structures; retain those structures in a frozen PubChem/COCONUT candidate snapshot | Can the database route rank the right constitutional isomer? |
| Formula isomer panel | Score truth against same-formula candidate isomers | Does a change fix the main class-2 failure mode? |
| Domain stress | Separate timsTOF natural-product and cross-instrument queries, plus scaffold/source holdouts | Are gains robust to chemistry and instrument shift? |
| Visible dummy test | Exact-match and output-format smoke check only | Does the pipeline read and write the required schema? |

Keep every query acquisition for a molecule together. Use a frozen validation split and a separate training/early-stopping split. Score one list per molecule. Report MRR@25, hit@1/5/25, candidate recall@25, formula accuracy, and per-source results. Run multiple seeds for learned rankers. A [participant report](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254) notes seed noise and same-formula isomers as a major error source; verify those observations locally.

## Execution phases and gates

### Phase 0 — reproducible project and data audit

- Initialize Git, add a user-owned GitHub remote, and ignore parquet data, tokens, model weights, and generated indexes.
- Record file hashes, schema, source counts, label quality (`precursor_error_ppm`), adduct distribution, and duplicate structure groups. Check the Kaggle data version because organizers announced a later training-data update.
- Implement metric-equivalent RDKit standardization and tautomer/InChIKey14 deduplication with the pinned version.
- **Gate:** reproducible metrics and no use of the dummy test for model selection.

### Phase 1 — strong CPU baseline

- The first `src/retrieve.py` implementation now streams the reference parquet and prunes comparisons by neutral precursor mass and ion mode. Benchmark this pass on Kaggle and tune the mass window, intensity floor, and fragment/neutral-loss blend.
- If profiling shows the streaming pass misses the 9-hour budget, build a persistent compact spectral index with precursor/adduct bins and sparse peak postings; keep peak arrays and labels in columnar storage.
- Tune intensity floors and top-N/window filters; compare multiple tolerances, square-root intensity, direct fragments, and neutral losses.
- Aggregate spectra across collision energies and adducts at molecule level, rank by InChIKey14, and produce the first valid Kaggle submission notebook.
- **Gate:** measured class-1 and class-2 holdout MRR/candidate recall, plus offline inference under the 9-hour notebook limit.

### Phase 2 — candidates for absent library spectra

- Build versioned, rule-compliant PubChem/COCONUT candidate tables with SMILES, exact mass, formula, standardized key, and fingerprints; store as Kaggle input artifacts.
- Infer candidate formulas/mass windows from precursor plus adduct, including water-loss adducts. Keep several plausible formulas when uncertain.
- Rank same-formula isomers with structure-aware fragment explanations and predicted fingerprints. Fragment evidence is a higher priority than extra database membership flags.
- **Gate:** improved class-2 candidate recall@25 and MRR on held-out structures, without class-1 regression.

### Phase 3 — learned reranker and GPU model

- Generate out-of-fold candidate lists and pairwise features. Train a CPU GBDT ranker with hard same-formula negatives; calibrate route scores on disjoint folds.
- Use Kaggle RTX PRO 6000 for a compact spectrum-to-fingerprint model after the CPU baseline is measured. Train on structure-grouped folds, checkpoint each run, and use mixed precision.
- Blend components only when they improve multi-seed held-out MRR and same-formula isomer ranking.
- **Gate:** meaningful gains on at least two relevant validation slices; otherwise retain the simpler model.

### Phase 4 — Kaggle packaging and final selection

- GitHub is the source of truth. Pin a Git SHA, upload the code snapshot to Kaggle before execution, and attach competition data plus versioned candidate/model artifacts. The committed notebook has internet disabled, so it cannot clone GitHub at runtime.
- Run a training notebook on Kaggle to produce weights/indexes, then an inference notebook using those frozen artifacts. Make each committed run fit the 9-hour limit.
- Write `submission.csv` with exactly one row per hidden `molecule_id`, up to 25 valid unique connectivity candidates, and no nulls. Keep runtime and provenance logs.
- Use the public leaderboard sparingly as a sanity signal; choose the final version by held-out validation and robustness across chemistry slices.

## Compute budget

| Resource | Assigned work |
|---|---|
| MacBook Pro M3, 8 GB | Code, docs, small batches, streamed metadata audits, Git/GitHub; avoid full in-memory parquet loads |
| Kaggle CPU | Full indexing, retrieval, feature generation, GBDT ranker, candidate tables, final notebook |
| Kaggle RTX PRO 6000, 30 h/week | Fingerprint-model pilots and final neural training after baseline gates; reserve time for inference profiling |

Google Cloud Run is excluded. Keep artifacts on Kaggle as versioned notebook outputs or private Kaggle datasets; keep code and configs on GitHub.

## Immediate next implementation milestone

Run `prepare_validation.py` for both structure-held-out and natural-product source-held-out splits on Kaggle, score the retrieval baseline, and profile its memory/runtime. This reveals whether the limiting factor is reference retrieval, candidate coverage, or constitutional-isomer ranking before spending GPU time.
