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

## Updated direction after the first scored submission (1 October 2026)

The first hidden-set public score was **0.162**. Treat it as evidence that the current system is mostly harvesting the library-match slice, not as proof of the exact hidden-test class distribution. A public four-channel reimplementation estimates the class-1 contribution at 0.162; its reported full system scored 0.328, while a public analog-propagation notebook reports 0.335. These are participant leaderboard reports, so use them as a strong research lead rather than a promised score.

The important architecture change is to rank a shared candidate pool with multiple evidence channels, instead of preserving the spectral list and only appending COCONUT predictions into vacant slots. The old fill-only blend did not work as intended: the spectral retriever emits 25 candidates even when its evidence is weak, leaving no room for the database list. A mass-shifted analog channel is now implemented and improves an independent 250-structure strict holdout. See [`reports/score_0162_research_and_next_steps.md`](reports/score_0162_research_and_next_steps.md) for sources and the experiment sequence.

## Validation that predicts the hidden set

| Slice | Construction | Question |
|---|---|---|
| Library-known natural products | Natural-product spectra from `enveda-np-examples`, RIKEN, GNPS, and MassBank as queries; exclude identical acquisitions but retain independent reference spectra for the compound | Can library search recover a known natural product across instruments? |
| Database-known, spectra absent | Hold out every spectrum for selected InChIKey14 structures; retain those structures in a frozen PubChem/COCONUT candidate snapshot | Can the database route rank the right constitutional isomer? |
| Formula isomer panel | Score truth against same-formula candidate isomers | Does a change fix the main class-2 failure mode? |
| Domain stress | Separate timsTOF natural-product and cross-instrument queries, plus scaffold/source holdouts | Are gains robust to chemistry and instrument shift? |
| Visible dummy test | Exact-match and output-format smoke check only | Does the pipeline read and write the required schema? |

Keep every query acquisition for a molecule together. Use a frozen validation split and a separate training/early-stopping split. Score one list per molecule. Report MRR@25, hit@1/5/25, candidate recall@25, formula accuracy, and per-source results. Run multiple seeds for learned rankers. A [participant report](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254) notes seed noise and same-formula isomers as a major error source; verify those observations locally.

### Local baseline measurements (1 October 2026)

- Source holdout: 1,184 spectra from 250 natural-product structures, excluding `enveda-np-examples` from retrieval; MRR@25 0.9261, hit@1 0.880, hit@5 0.976, hit@25 0.996. This measures retrieval when another library still contains the labeled structure.
- Strict metric-structure holdout: 2,096 spectra from 278 structures; all 279 raw InChIKey14 aliases mapping to those metric keys were excluded from reference retrieval. Library-only MRR@25 and hit@25 were both 0. This is expected because every candidate is drawn from the remaining training structures, so it directly demonstrates that a spectral-library retriever cannot solve the unseen-structure classes by itself.
- The strict split's metric-key map is cached and checkpointed; its first local build canonicalized 275,810 unique source keys in about 9 minutes on four CPU workers. Retrieval completed in about 5 minutes. The Kaggle notebook uses CPU; GPU allocation does not accelerate these steps.
- Internal database-known holdout: built a 273,681-structure candidate pool from training structure metadata, removed all spectra for each selected target, and queried 547 `enveda-180` structures (3,336 spectra). Exact-mass ranking achieved MRR@25 0.1438, hit@1 0.0494, hit@5 0.2395, hit@25 0.5850. A single-bond fragment reranker with a nominal 5 ppm precursor window (0.01 Da minimum radius) improved this to MRR@25 0.3129, hit@1 0.1993, hit@5 0.4406, hit@25 0.7550. True precursor masses were within 5 ppm for 99.6% of this slice.
- A CPU HistGradientBoosting reranker trained on candidate fragment/mass/structure features. Across three molecule-grouped seeds, weighted validation MRR@25 averaged 0.5096 versus 0.3566 for fragment-only ranking; hit@1 averaged 0.3817 versus 0.2456, hit@5 0.6568 versus 0.4970, and hit@25 0.8787 versus 0.7633. Each validation fold contained 101–123 molecule IDs; the same molecule may occur in different seeds' validation folds.
- The internal pool is derived from structures in the competition training set. This is a useful closed-world ranking benchmark, not an independent COCONUT/public-database result; rerun the same holdout with the frozen external candidate snapshot before treating the fragment gain as a class-2 estimate.
- External COCONUT pilot: used the October 2026 CC0 CSV-lite snapshot and held out all 46 structures overlapping COCONUT and `enveda-180` (309 spectra). These query IDs do not overlap the 547 internal development queries. The candidate table contains 8,851 COCONUT structures in 50 ppm windows around the target precursor masses. Exact-mass ranking scored MRR@25 0.2876, hit@1 0.1522, hit@5 0.4130, hit@25 0.7609; the 5 ppm fragment reranker scored 0.4051, 0.2609, 0.5652, and 0.8261. Three internal-seed models averaged 0.5631 MRR@25, 0.4130 hit@1, 0.7536 hit@5, and 0.9130 hit@25 on those same 46 structures. Treat this as a small pilot because it contains only 46 targets.
- **Mass-shifted analog channel (strict external structure holdout):** retrieved up to 128 high-similarity library analogs within ±200 Da and transferred their Morgan-fingerprint similarity to 33,443 mass-filtered COCONUT structures for 250 natural-product queries. All held-out target structures and raw-key aliases were excluded from the reference scan. COCONUT contained the truth for 248/250 targets (99.2% recall). Analog-only ranking reached MRR@25 0.5197, hit@1 0.3508, hit@5 0.7379, and hit@25 0.9315 on the 248 covered targets. On shared candidate rows, the fragment-only ranker scored MRR 0.2638; fragment+analog reached 0.5212 (+0.2574 absolute, nearly 2×), with hit@1 0.3629 and hit@25 0.9194. This is the strongest measured improvement so far for structures missing from the spectral library.
- The same 15-feature ranker was evaluated with molecule-grouped five-fold out-of-fold predictions over 547 internal development molecules: fragment-only MRR 0.4750 versus multichannel MRR 0.7961. This internal gain is optimistic relative to the independent external result and should not be treated as an expected leaderboard score.
- **Spectral/database route integration:** source-known retrieval gives MRR 0.9261, while the independent COCONUT route reaches 0.5212 on the same broad source slice after target structures are excluded from analog references. The old fill-only blend always retained spectral candidates first because retrieval returns a full top-25 list. A competition-key-deduplicated top-two spectral margin now drives route ordering: below 0.03 the database list leads; otherwise spectral leads. Source-known and strict-structure margins overlap, so this is a heuristic requiring leaderboard feedback, not a validated class detector. Updated metric-key-deduplicated diagnostics are in the research report.

### Local multichannel measurements (2 October 2026)

- The final inference notebook fits the 15-feature CPU ranker and generates fragment plus mass-shifted analog features for mass-filtered COCONUT candidates.
- `artifacts/kaggle_upload/CASMI26_Ranker_Training_Features.zip` is the private Kaggle upload artifact. It contains the merged 117,952-row table and README; upload as a new version of the existing private ranker-features dataset through the Kaggle website.
- The margin-based route ordering is a starting heuristic. Compare the public score and independent holdouts before further tuning; do not choose it based on the visible dummy test.

These are CPU runs on the MacBook Pro M3. The local strict holdout, source holdout, candidate construction, reranking, and conservative blend have all run without GPU or Cloud Run. Keep the RTX allocation unused until a neural spectrum-to-fingerprint experiment beats the CPU system on multiple independent molecule-grouped folds. The 46-target COCONUT sample and 250-target source slice are not enough to claim hidden-set performance.

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

- GitHub is the source of truth. Pin a Git SHA, create the source snapshot with `scripts/package_kaggle_source.py`, and upload it through the Kaggle website as a private Dataset. Attach competition data plus versioned candidate/model artifacts in the notebook's Input panel. Do not use Kaggle CLI; the committed notebook has internet disabled and cannot clone GitHub at runtime.
- Use `notebooks/03_final_inference.ipynb` for the website-based run: it fits the CPU reranker from the updated private 15-feature input, builds test-mass-filtered COCONUT candidates, scores fragments and mass-shifted analogs, runs spectral retrieval with confidence diagnostics, routes candidate order, and validates `submission.csv`. Keep the run under the 9-hour limit.
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

Implement and evaluate the candidate-level evidence fusion locally first. Start with the mass-shifted analog channel on the existing source and structure holdouts; add the public CASMI26 fingerprint checkpoints and candidate fingerprint table as Kaggle inputs only after the local channel has demonstrated value. Keep the existing spectral route as one evidence feature, not a protected prefix. Use the website-based notebook to evaluate the integrated system on the hidden rerun after local validation is stable. The visible dummy test remains limited to schema/runtime checks.
