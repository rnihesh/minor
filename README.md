# Robustness-First Speech Emotion Recognition

Speech Emotion Recognition (SER) project using TensorFlow with:

- multi-dataset ingestion (`RAVDESS`, `CREMA-D`, `TESS`, `SAVEE`)
- dual evaluation protocol (`random_stratified` + `speaker_independent`)
- configurable feature bundles (MFCC-only or robust stacked features)
- model suite (`baseline`, `attention`, `lightweight`)
- benchmark reporting against top-6 literature references

## Key Upgrades

- Unified metadata schema for all datasets:
  - `dataset_id`, `speaker_id`, `emotion_id`, `sr`, `duration`, `path`
- Dual protocol split engine:
  - `random`: paper-style random stratified split
  - `speaker`: strict speaker-held-out split
  - `dual`: runs both
- Robust feature pipeline:
  - MFCC + delta + delta-delta + log-mel + ZCR
  - optional waveform augmentation (noise/shift/speed/pitch)
  - optional SpecAugment masks
- Benchmark report generator:
  - per-protocol metrics
  - per-dataset metrics
  - macro-F1, weighted F1, UAR
  - drawback-mitigation table mapped to papers #1-#6

## Installation

```bash
uv sync
uv run python main.py verify-gpu
```

## CLI

### 1) Download open dataset pack

```bash
uv run python main.py download-datasets --pack open4
```

If a dataset cannot be downloaded automatically, the command prints manual placement instructions.

### 2) Train

```bash
uv run python main.py train \
  --datasets ravdess,crema_d,tess,savee \
  --protocol dual \
  --model-variant attention \
  --feature-bundle robust \
  --epochs 30 --batch-size 32 --use-focal-loss
```

Quick smoke run:

```bash
uv run python main.py train \
  --datasets ravdess \
  --protocol random \
  --model-variant lightweight \
  --feature-bundle mfcc \
  --epochs 1 --batch-size 16 --no-augmentation
```

### 3) Evaluate

```bash
uv run python main.py evaluate
```

Multi-dataset evaluation:

```bash
uv run python main.py evaluate \
  --datasets ravdess,crema_d,tess,savee \
  --protocol speaker
```

### 4) Benchmark vs first 6 papers

Using an existing run:

```bash
uv run python main.py benchmark --papers first6 --run-id <run_id>
```

Auto-train if no run is found:

```bash
uv run python main.py benchmark \
  --papers first6 \
  --protocol dual \
  --train-if-missing \
  --datasets ravdess,crema_d,tess,savee \
  --model-variant attention \
  --feature-bundle robust
```

Outputs are written to:

- `outputs/reports/*.md`
- `outputs/reports/*.json`

### 5) Predict

```bash
uv run python main.py predict --audio /path/to/sample.wav
```

`predict` auto-detects feature mode from the loaded model input shape (legacy MFCC or robust bundle).

## Model Variants

- `baseline`: CNN-LSTM
- `attention`: CNN + BiLSTM + MultiHeadAttention
- `lightweight`: SeparableConv1D + compact LSTM stack

## Testing

Run full tests (includes 1-epoch dual-protocol smoke training on synthetic mini data):

```bash
uv run --with pytest python -m pytest -q
```

Current test coverage includes:

- unified metadata/schema integrity
- speaker overlap checks for `speaker_independent` splits
- feature shape and NaN safety
- one-epoch smoke training for both protocols
- deterministic benchmark report rendering

## Core Files

- `src/config.py`: global config + dataclasses for feature/split/train/augmentation
- `src/datasets.py`: unified dataset scanners and metadata validation
- `src/splits.py`: random + speaker-independent split strategies
- `src/feature_extraction.py`: robust feature extraction and augmentations
- `src/model.py`: baseline/attention/lightweight model builders
- `src/train.py`: training pipeline + run summaries
- `src/evaluate.py`: metrics, confusion matrix, per-dataset evaluation
- `src/benchmark.py`: paper comparison and report generation
- `src/download_datasets.py`: open4 dataset download helper

## Literature Set Used in Benchmark

1. Ouyang (2025) - arXiv: https://arxiv.org/abs/2501.10666
2. Salian et al. (2021) - DOI: https://doi.org/10.1051/itmconf/20214003006
3. Zhao et al. (2018) - DOI: https://doi.org/10.1016/j.neucom.2017.10.005
4. Ullah et al. (2023) - DOI: https://doi.org/10.3390/s23136212
5. Madanian et al. (2023) - ScienceDirect: https://www.sciencedirect.com/science/article/pii/S2667305323000911
6. Bhanbhro et al. (2025) - DOI: https://doi.org/10.3390/signals6020022

## Notes

- Legacy RAVDESS-only APIs are still preserved for compatibility (`prepare_data`, `create_cnn_lstm_model`).
- Existing `archive/Actor_*` RAVDESS structure is still supported.
