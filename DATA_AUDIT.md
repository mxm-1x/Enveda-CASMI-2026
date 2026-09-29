# Local data audit — 29 September 2026

## Files inspected

| File | Observed rows | Columns | Parquet row groups | Local size |
|---|---:|---:|---:|---:|
| `train.parquet` | 2,539,608 | 18 | 21 | about 2.8 GB |
| `test.parquet` | 1,213 | 12 | 1 | about 4.6 MB |
| `sample_submission.csv` | 400 molecule rows | 2 | — | about 43 KB |

The visible test has 400 distinct `molecule_id` values, 1–9 spectra per molecule (median 3), 987 positive and 226 negative spectra, and all 1,213 rows report `instrument_type=timsTOF`. Its adduct counts are `[M+H]+` 959, `[M-H]-` 193, `[M+CH2O2-H]-` 31, `[M+Na]+` 22, `[M+NH4]+` 4, `[M+K]+` 2, and `[M+Cl]-` 2. These describe the **dummy file**, not the hidden set.

## Full training scan

| Source | Spectra |
|---|---:|
| enveda-180 | 1,153,785 |
| pluskal_ms2 | 527,581 |
| riken | 347,171 |
| gnps | 220,849 |
| massbank | 101,727 |
| mona | 92,416 |
| spectraverse | 50,933 |
| msdial | 40,765 |
| drug_plus | 2,545 |
| enveda-np-examples | 1,184 |
| masaryk | 652 |

The train file has 121 distinct adduct strings, including water-loss adducts. The `enveda-np-examples` count differs from the published data-page table, so confirm the current Kaggle input version before freezing splits or weights.

## Exact overlap check

A streamed scan compared full `ms2_mzs` and aligned `ms2_normalized_intensities` arrays. It used peak count plus first/last peak values only to shortlist, then required full array equality.

- 1,213/1,213 visible test spectra had one exact training match.
- 400/400 visible molecules had at least one match.
- All matching rows came from `enveda-180`.
- No molecule had conflicting InChIKey14 labels across its matched rows.

This is expected: the [Kaggle data page](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) says the downloadable test file is assembled from training examples and replaced by hidden test data during the notebook rerun. The [host explicitly confirms](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741404) the leaderboard scores that hidden set. The overlap result therefore checks ingestion and lookup correctness; it is not a predictive validation result.
