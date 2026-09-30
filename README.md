# Enveda CASMI 2026

Rank up to 25 candidate SMILES per molecule from MS/MS spectra. The score is molecule-level MRR@25 using RDKit tautomer-canonicalized InChIKey14 connectivity.

Start with the [data-grounded strategy](PLAN.md), then the [pipeline architecture](PIPELINE_ARCHITECTURE.md) and [local data audit](DATA_AUDIT.md). The strategy covers validation, candidate retrieval, database ranking, neural training gates, Kaggle packaging, and the available compute budget.

## Key local finding

The added folder `enveda-CASMI26-molecule-id-mass-spectra/` contains 2,539,608 labeled training spectra, 1,213 visible test spectra, and a 400-molecule sample submission. A complete streamed audit found that every visible test spectrum occurs exactly once in training, all in `enveda-180`. The competition [data page](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) and [host clarification](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741404) explain that this is a dummy test file. Kaggle replaces it with hidden natural-product-like spectra when the notebook is rerun. Use the local test for schema and runtime checks only.

## Code added

The code includes streaming retrieval, spectrum preprocessing, RDKit metric keys, structure-grouped folds, source- and structure-held-out validation, a COCONUT CSV/ZIP candidate-table builder, a validation scorer, overlap auditing, and submission validation. The [CPU validation notebook](notebooks/01_cpu_validation.ipynb) runs both holdouts offline. RDKit 2026.03.3 is required to match competition scoring.

## Current results and decision

The retrieval baseline scored MRR@25 **0.9261** on a 250-structure natural-product source holdout (hit@1 0.880; hit@5 0.976; hit@25 0.996). This is a known-structure retrieval setting: spectra from `enveda-np-examples` were held out while other library spectra for the same structures remained.

On a stricter split, 278 metric structures and all 2,096 associated spectra were removed from the reference. The library-only retriever scored **0.000 MRR@25** as expected when the true structures are absent from its candidate pool. This is the key next step: add a frozen public natural-product candidate database and score structures that have no reference spectra. The competition describes hidden examples spanning public spectral-library compounds, known structures without public spectra, and novel structures; a library-only method cannot cover all three classes.

## Local development and Kaggle

Use the MacBook for code, small checks, and Git. The full structure-key map took about nine minutes locally on four CPU workers and is cached for reuse. Kaggle CPU is appropriate for the current retrieval and database pipeline. Save the limited GPU allocation for a learned spectrum-to-fingerprint model after database candidate recall is measured.

Kaggle submissions must run in a notebook with internet disabled and produce `submission.csv`. Upload the versioned source snapshot as a private Kaggle Dataset, attach the competition data and pinned RDKit wheel dataset, then run [01_cpu_validation.ipynb](notebooks/01_cpu_validation.ipynb). No Kaggle CLI or Google Cloud Run is part of this workflow.

To refresh the upload snapshot after committing code:

```bash
python3 scripts/package_kaggle_source.py
```

The package is staged at `/private/tmp/casmi-source` and copied into `data/kaggle-source-upload/` for the Kaggle Dataset version. The project uses no Kaggle CLI and excludes Google Cloud Run.

After downloading the official COCONUT CSV-lite archive, attach it as a private Kaggle Dataset and run `src/prepare_candidates.py` on Kaggle CPU to create a compact Parquet table with canonical SMILES, formula, exact mass, and the competition metric key. It deduplicates structures by InChIKey14 and writes a provenance manifest beside the table.
