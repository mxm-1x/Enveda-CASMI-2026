# Enveda CASMI 2026

Rank up to 25 candidate SMILES per molecule from MS/MS spectra. The score is molecule-level MRR@25 using RDKit tautomer-canonicalized InChIKey14 connectivity.

Start with the [data-grounded strategy](PLAN.md), then the [pipeline architecture](PIPELINE_ARCHITECTURE.md) and [local data audit](DATA_AUDIT.md). The strategy covers validation, candidate retrieval, database ranking, neural training gates, Kaggle packaging, and the available compute budget.

## Key local finding

The added folder `enveda-CASMI26-molecule-id-mass-spectra/` contains 2,539,608 labeled training spectra, 1,213 visible test spectra, and a 400-molecule sample submission. A complete streamed audit found that every visible test spectrum occurs exactly once in training, all in `enveda-180`. The competition [data page](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) and [host clarification](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741404) explain that this is a dummy test file. Kaggle replaces it with hidden natural-product-like spectra when the notebook is rerun. Use the local test for schema and runtime checks only.

## Code added

The code includes streaming retrieval, spectrum preprocessing, RDKit metric keys, structure-grouped folds, source-, structure-, and database-held-out validation, a COCONUT CSV/ZIP candidate-table builder, exact-mass and bond-cleavage fragment candidate rankers, a CPU gradient-boosting reranker, fill-only prediction blending, a validation scorer, overlap auditing, and submission validation. The [CPU validation notebook](notebooks/01_cpu_validation.ipynb) runs the library holdouts; [the database validation notebook](notebooks/02_database_validation.ipynb) builds the external candidate table and measures the COCONUT route. RDKit 2026.03.3 is required to match competition scoring.

## Current results and decision

The retrieval baseline scored MRR@25 **0.9261** on a 250-structure natural-product source holdout (hit@1 0.880; hit@5 0.976; hit@25 0.996). This is a known-structure retrieval setting: spectra from `enveda-np-examples` were held out while other library spectra for the same structures remained.

On a stricter split, 278 metric structures and all 2,096 associated spectra were removed from the reference. The library-only retriever scored **0.000 MRR@25** as expected when the true structures are absent from its candidate pool.

A closed-world database holdout used 547 `enveda-180` molecules (3,336 spectra) and 273,681 candidate structures from training metadata. Exact-mass ranking scored **0.1438 MRR@25** and **0.5850 hit@25**. A lightweight one-bond fragment reranker improved this to **0.3129 MRR@25** and **0.7550 hit@25**. Across three molecule-grouped seeds, the CPU HistGradientBoosting reranker averaged **0.5096 MRR@25** and **0.8787 hit@25** on validation folds. This candidate pool is derived from training structures, so it is a development benchmark rather than an independent database result.

That external check is now complete as a pilot: on the October 2026 COCONUT snapshot, 46 overlapping `enveda-180` structures (309 spectra) were held out while retaining their structures among 8,851 mass-windowed COCONUT candidates. These query IDs are disjoint from the internal training queries. Exact-mass ranking scored **0.2876 MRR@25** / **0.7609 hit@25**; the fragment reranker scored **0.4051** / **0.8261**; three CPU-trained HistGradientBoosting models averaged **0.5631** / **0.9130**. The external sample is small, so use these results as a promising pilot and validate on a broader independent candidate set before final inference.

On a common 250-molecule source holdout, COCONUT-only candidates with an internal-model ensemble scored **0.2745 MRR@25** (candidate recall@25 0.724), while spectral retrieval scored **0.9261**. The final CPU model, fit on 547 disjoint training groups, scored **0.2628 MRR@25** alone. The implemented `src/blend_predictions.py` preserves spectral rank and fills open positions only; the final model blend scored **0.92634 MRR@25** and raised hit@25 from **0.996 to 1.000** on that holdout. This small gain supports using database candidates as a fallback, not ranking them ahead of spectral matches.

## Local development and Kaggle

The full CPU validation and model development were run on the MacBook Pro M3 with 8 GB RAM; the 2.5M-row train parquet was streamed rather than loaded into memory. Use Kaggle only for the website-based hidden-test notebook run. Save the limited GPU allocation for a learned spectrum-to-fingerprint model after a broader database candidate benchmark.

Kaggle submissions must run in a notebook with internet disabled and produce `submission.csv`. For final inference, use [03_final_inference.ipynb](notebooks/03_final_inference.ipynb). Attach the competition data, versioned source snapshot, COCONUT ZIP, private ranker feature table, and the pinned RDKit wheel dataset if Kaggle's installed RDKit is not 2026.03.3. No Kaggle CLI or Google Cloud Run is part of this workflow.

To refresh the upload snapshot after committing code:

```bash
python3 scripts/package_kaggle_source.py
```

The package is staged at `/private/tmp/casmi-source` and copied into `data/kaggle-source-upload/` for the Kaggle Dataset version. The small, private training feature input is in `data/kaggle-ranker-training/ranker_training_features.parquet` (also packaged as `data/CASMI26_Ranker_Training_Features.zip`). Suggested Kaggle Dataset names are `casmi26-source-code` and `casmi26-ranker-training-features`. The project uses no Kaggle CLI and excludes Google Cloud Run.

After downloading the official COCONUT CSV-lite archive, attach it as a private Kaggle Dataset and run `src/prepare_candidates.py` on Kaggle CPU to create a compact Parquet table with canonical SMILES, formula, exact mass, and the competition metric key. It deduplicates structures by InChIKey14 and writes a provenance manifest beside the table.
