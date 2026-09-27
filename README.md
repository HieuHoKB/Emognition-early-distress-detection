# Early-Warning Distress Detection — Research Code

This repository contains the implementation of the early-warning distress
detection pipeline described in the thesis *"Toward Reliable Early Warning
of Psychological Distress"*. The code trains a participant-independent
tabular detector on Samsung Watch features and optionally adds a
second-stage distress-versus-surprise gate.

| Component             | Evaluation setting                              | Key result                                        | Caveat                                                                        |
| --------------------- | ----------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------- |
| Primary HGBT detector | 42 admissible LOSO folds; 30% session prefix    | Macro-F1 0.736; distress recall 0.850             | Surprise false-alarm rate 1.000                                               |
| Two-stage HGBT gate   | Core non-distress + surprise challenge sessions | Confirmed recall 0.707; surprise misrouting 0.238 | Same surprise folds informed selection and reporting; not external validation |

These are exploratory model-selection summaries on one laboratory archive. They are not clinical-performance estimates or evidence of deployment readiness.

## Reproducibility status

This repository provides the public implementation, experiment
configuration, seeds, and synthetic smoke test for the study.

The original sensor data and derived feature table are not redistributed.
They are subject to the source dataset's access and data-handling terms.
Qualified academic reviewers may contact the author to request
verification materials where permitted.


## What is in this repository

| File | Role |
|------|------|
| `config.py` | Loads `experiment_01_baseline.yaml` into a flat dict. |
| `prepare_labels.py` | Curates the questionnaire-driven session labels (Section 3.4). |
| `prepare_features.py` | Computes the 108-dimensional prefix feature table (Section 3.5). |
| `research_pipeline.py` | Thin wrapper that builds the prefix feature artifacts. |
| `model_backends.py` | Single chokepoint for the four tabular backends (logreg, hgbt, catboost, lightgbm). |
| `train.py` | Primary-detector LOSO trainer. |
| `train_with_gate.py` | Second-stage gate benchmark. |
| `two_model_gate.py` | Four-state routing logic (Section 3.7, Equations 15-18). |
| `experiment_01_baseline.yaml` | The single experiment configuration. |
| `seeds.json` | Master seed inventory. |
| `seeds.csv` | Flat form of `seeds.json`. |
| `requirements.txt` | Pinned software versions. |
| `test_smoke.py` | Smoke test with synthetic data. |

## Software environment

The software versions used to produce the reported results are listed in
`requirements.txt`. A reference environment (Section 4.5 of the thesis)
runs the tabular backends on a single CPU-only machine.

```
Python 3.13.7
numpy 2.3.5
pandas 2.2.3
scikit-learn 1.8.0
catboost 1.2.10
lightgbm 4.6.0
torch 2.11.0
```

## How to use

### 1. Prepare the questionnaire manifest

The manifest must contain at least these columns:

```
session_id, stimulus_category, anger, fear, disgust, sadness,
valence, motivation
```

The script writes a curated `session_label_manifest.csv` next to the input
questionnaire.

```bash
python prepare_labels.py \
    --questionnaire data/raw/session_questionnaire.csv \
    --output      artifacts/curation/session_label_manifest.csv
```

### 2. Resample the raw sensor files to a shared 10 Hz grid

The resampled directory must contain one CSV per non-baseline session with
the columns:

```
timestamp, heartRate, PPInterval, BVPProcessed,
acc_mag, gyr_mag, rot_mag
```

A typical resampling pipeline reads the raw Samsung Watch exports, drops
duplicates, and linearly interpolates onto a 10 Hz grid while remasking
interior gaps longer than 2.0 s. The exact resampling script is data-source
specific and is therefore not redistributed here.

### 3. Build the prefix feature table

```bash
python research_pipeline.py \
    --resampled-dir   data/resampled/ \
    --label-manifest  artifacts/curation/session_label_manifest.csv \
    --output-dir      artifacts/dense-prefix/
```

This writes `artifacts/dense-prefix/prefix_feature_table.csv` with one row
per (session, prefix ratio) pair and 108 derived features.

### 4. Run the primary detector

```bash
python train.py \
    --resampled-dir   data/resampled/ \
    --label-manifest  artifacts/curation/session_label_manifest.csv \
    --output-dir      artifacts/baseline/
```

This writes `summary.csv` and `loso_predictions.csv` for the primary
detector across all (backend, prefix ratio) combinations.

### 5. Run the second-stage gate

```bash
python train_with_gate.py \
    --resampled-dir        data/resampled/ \
    --label-manifest       artifacts/curation/session_label_manifest.csv \
    --primary-predictions  artifacts/baseline/loso_predictions.csv \
    --output-dir           artifacts/baseline/gate/
```

This writes `gate_results.csv` with the per-fold best gate
configuration and its routing metrics.

## What is NOT in this repository

* **Raw Samsung Watch sensor data** are not redistributed in this repository. Users should obtain the source dataset directly from its original provider and comply with its access terms, licence, and applicable data-handling requirements.
* **Resampling scripts** are data-source specific and live in the
  author's private working repository.
* **Author-trained checkpoints** are not redistributed.

Reviewers who want to verify a single number can request the resampled
feature table from the author and rerun `train.py` + `train_with_gate.py`
against the public configuration and seeds.

## License

MIT License

## Citation

If you use this code, please cite:

Ho, T. H. (2026). Feasibility of Early Distress Warning From
Partial Wearable Observations in the Emognition Dataset.
Presented at the 11th International Conference on the Development
of Biomedical Engineering (BME11). Unpublished manuscript / conference presentation.
