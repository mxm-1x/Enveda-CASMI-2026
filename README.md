# Enveda CASMI 2026

Rank up to 25 candidate SMILES per molecule from MS/MS spectra. The score is molecule-level MRR@25 using RDKit tautomer-canonicalized InChIKey14 connectivity.

Start with the [data-grounded strategy](PLAN.md), then the [pipeline architecture](PIPELINE_ARCHITECTURE.md) and [local data audit](DATA_AUDIT.md). The strategy covers validation, candidate retrieval, database ranking, neural training gates, Kaggle packaging, and the available compute budget.

## Key local finding

The added folder `enveda-CASMI26-molecule-id-mass-spectra/` contains 2,539,608 labeled training spectra, 1,213 visible test spectra, and a 400-molecule sample submission. A complete streamed audit found that every visible test spectrum occurs exactly once in training, all in `enveda-180`. The competition [data page](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) and [host clarification](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741404) explain that this is a dummy test file. Kaggle replaces it with hidden natural-product-like spectra when the notebook is rerun. Use the local test for schema and runtime checks only.

## Code added

The code now includes streaming retrieval, spectrum preprocessing, RDKit metric keys, structure-grouped folds, validation split preparation, a validation scorer, overlap auditing, and submission validation. [The Kaggle baseline notebook](notebooks/00_retrieval_baseline.ipynb) discovers attached inputs and runs the current retrieval pipeline offline. The retrieval method is a first baseline and still needs grouped validation and tuning. RDKit 2026.03.3 is required; the notebook expects its offline Kaggle wheel input.

No trained model, validation score, GitHub push, or Kaggle submission is claimed yet. This directory is not currently a Git repository.

## First commands

The local Mac can run the overlap audit with the available PyArrow installation:

```bash
python3 src/audit_overlap.py \
  --train enveda-CASMI26-molecule-id-mass-spectra/train.parquet \
  --test enveda-CASMI26-molecule-id-mass-spectra/test.parquet
```

For validation and metric-equivalent submission checks, run the Kaggle notebook with the pinned offline RDKit input attached. Its metadata template is [kernel-metadata.example.json](kaggle/kernel-metadata.example.json); replace the account and dataset slugs before upload.

After accepting the competition rules and configuring the Kaggle CLI, commit source changes and stage a clean snapshot with:

```bash
python3 scripts/package_kaggle_source.py
```

Upload `/private/tmp/casmi-source` as a private Kaggle Dataset, then attach it with the competition input and the pinned offline RDKit dataset. The validation notebook is [01_cpu_validation.ipynb](notebooks/01_cpu_validation.ipynb); it measures source-held-out natural-product retrieval and a small structure holdout before we spend GPU time.

The project uses the local MacBook for development and Kaggle CPU/GPU for data processing and training. Google Cloud Run is excluded. A committed competition notebook must run offline within nine hours; code from GitHub must be uploaded to Kaggle before execution.
