# Execution Guide (From Scratch)

This file explains exactly how to run this project end-to-end, even if you are setting it up for the first time.

## 1) What This Project Does

This repo trains and evaluates a Speech Emotion Recognition (SER) system with:
- Multi-dataset ingestion: `RAVDESS`, `CREMA-D`, `TESS`, `SAVEE`
- Two evaluation protocols: `random_stratified` (paper-style), `speaker_independent` (strict actor/speaker holdout)
- Multiple model variants: `baseline`, `attention`, `lightweight`
- Benchmark report against the selected top-6 papers

## 2) Prerequisites

Install these first:
- Git
- Python `3.11` or `3.12`
- `uv` (package/env manager)

Install `uv`:
- Follow: https://docs.astral.sh/uv/getting-started/installation/

## 3) Clone + Setup Environment

If you do not have the repo yet:

```bash
git clone <your-repo-url>
cd minor
```

If you already have the repo:

```bash
cd /path/to/minor
```

Install dependencies:

```bash
uv sync
```

Optional GPU check:

```bash
uv run python main.py verify-gpu
```

## 4) Dataset Setup (Important)

### 4.1 Automatic download (recommended first)

```bash
uv run python main.py download-datasets --pack open4
```

This tries to download:
- `ravdess`
- `crema_d`
- `tess`
- `savee`

The command prints per-dataset status:
- `downloaded`: ready
- `manual_required`: you must place files manually

### 4.2 Manual dataset placement (if needed)

Use these target folders:
- `data/ravdess/` OR existing `archive/Actor_01...Actor_24` (already supported)
- `data/crema_d/`
- `data/tess/`
- `data/savee/`

Keep WAV filenames/folder patterns as close to original dataset structure as possible.

### 4.3 Sanity check which datasets are discovered

```bash
uv run python - <<'PY'
from src.datasets import load_unified_metadata, metadata_summary
md = load_unified_metadata(['ravdess','crema_d','tess','savee'], strict=False)
print(metadata_summary(md))
PY
```

If only `ravdess` appears, other datasets are still missing or not parsed correctly.

## 5) Run Tests

Run full automated tests:

```bash
uv run --with pytest python -m pytest -q
```

What tests cover:
- Data schema integrity
- Split correctness (`speaker_independent` has zero speaker overlap)
- Feature shape/NaN safety
- 1-epoch smoke training
- Benchmark report determinism

## 6) Training Commands

### 6.1 Quick smoke run (fast)

Use this to confirm pipeline works:

```bash
uv run python main.py train \
  --datasets ravdess \
  --protocol random \
  --model-variant lightweight \
  --feature-bundle mfcc \
  --epochs 1 \
  --batch-size 16 \
  --no-augmentation
```

### 6.2 Main robustness-first training (recommended)

```bash
uv run python main.py train \
  --datasets ravdess,crema_d,tess,savee \
  --protocol dual \
  --model-variant attention \
  --feature-bundle robust \
  --epochs 30 \
  --batch-size 32 \
  --use-focal-loss
```

### 6.3 Useful options

- `--protocol random|speaker|dual`
- `--model-variant baseline|attention|lightweight`
- `--feature-bundle mfcc|robust`
- `--no-augmentation` (disable augmentation)
- `--no-class-weight` (disable class weighting)

### 6.4 Training outputs

After training, check:
- Models: `models/*_best.keras`, `models/*_final.keras`
- Logs: `outputs/logs/`
- Run summaries (important): `outputs/runs/<run_id>.json`
- Plots: `outputs/*history.png`, `outputs/*confusion_matrix.png`, `outputs/*per_class_metrics.png`

## 7) Evaluation Commands

Evaluate latest available model:

```bash
uv run python main.py evaluate
```

Evaluate with explicit datasets/protocol:

```bash
uv run python main.py evaluate \
  --datasets ravdess,crema_d,tess,savee \
  --protocol speaker
```

Evaluate a specific model file:

```bash
uv run python main.py evaluate --model models/<model_file>.keras
```

## 8) Benchmark Report (Top-6 Papers)

Generate benchmark from an existing run id:

```bash
uv run python main.py benchmark --papers first6 --run-id <run_id>
```

If no run summary exists, auto-train and benchmark:

```bash
uv run python main.py benchmark \
  --papers first6 \
  --protocol dual \
  --train-if-missing \
  --datasets ravdess,crema_d,tess,savee \
  --model-variant attention \
  --feature-bundle robust
```

Benchmark outputs:
- Markdown report: `outputs/reports/*.md`
- JSON payload: `outputs/reports/*.json`

## 9) Inference / Prediction

Predict emotion for a WAV file:

```bash
uv run python main.py predict --audio /absolute/path/to/file.wav
```

Model auto-selection:
- If `--model` is omitted, the latest best model is used.

## 10) Team Workflow (Recommended)

For each teammate:
1. Pull latest code.
2. Run `uv sync`.
3. Run `uv run --with pytest python -m pytest -q`.
4. Run one quick smoke train command.
5. Run main training for assigned experiment.
6. Share `run_id`, key metrics (accuracy, macro-F1, UAR), and benchmark report path.

## 11) Troubleshooting

`download-datasets` shows `manual_required`:
- This is expected for some sources. Download manually and place files in the required `data/<dataset>/` folder.

Very low accuracy in 1 epoch:
- Expected for smoke runs. Use longer training (20-50 epochs) and robust feature bundle.

Out-of-memory or very slow training:
- Reduce batch size, e.g. `--batch-size 8` or `16`.
- Try `--model-variant lightweight` first.

Only RAVDESS is being used:
- Run the metadata sanity check in section 4.3 and verify file placement.

## 12) One-Command Help

```bash
uv run python main.py --help
uv run python main.py train --help
uv run python main.py evaluate --help
uv run python main.py benchmark --help
uv run python main.py download-datasets --help
```
