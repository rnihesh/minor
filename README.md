# Speech Emotion Recognition Using CNN-LSTM

A deep learning system for recognizing human emotions from speech audio using a hybrid CNN-LSTM architecture, trained on the RAVDESS dataset. Optimized for Apple Silicon (M4 Pro) with Metal GPU acceleration.

## Overview

This project implements an 8-class emotion classifier that can recognize:
- Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

## Project Structure

```
minor/
├── archive/                    # RAVDESS dataset
│   └── Actor_01/ ... Actor_24/
├── src/
│   ├── __init__.py
│   ├── config.py              # Hyperparameters & paths
│   ├── data_loader.py         # Load & preprocess dataset
│   ├── feature_extraction.py  # MFCC extraction with Librosa
│   ├── model.py               # CNN-LSTM architecture
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Metrics & visualization
│   └── predict.py             # Inference on new audio
├── models/                    # Saved trained models
├── outputs/                   # Training logs, plots
├── requirements.txt
├── main.py                    # Entry point
└── README.md
```

## Installation

### Using uv (recommended)

```bash
# Sync dependencies (creates venv automatically)
uv sync

# Verify Metal GPU
uv run python main.py verify-gpu
```

Expected output:
```
TensorFlow version: 2.x.x
Physical devices:
  CPU: /physical_device:CPU:0
  GPU: /physical_device:GPU:0
Metal GPU acceleration: ENABLED (1 GPU(s))
```

## Usage

### Train Model

```bash
uv run python main.py train
```

With custom parameters:
```bash
uv run python main.py train --epochs 50 --batch-size 16
```

### Evaluate Model

```bash
uv run python main.py evaluate
```

With specific model:
```bash
uv run python main.py evaluate --model models/emotion_cnn_lstm_best.keras
```

### Predict Emotion

```bash
uv run python main.py predict --audio archive/Actor_01/03-01-01-01-01-01-01.wav
```

Output:
```
Predicted: neutral (confidence: 87.3%)

All probabilities:
  neutral    ██████████████████████████░░░░  87.3%
  calm       ███░░░░░░░░░░░░░░░░░░░░░░░░░░░   5.2%
  ...
```

## Model Architecture

### CNN-LSTM Hybrid

```
Input: (130, 40) - MFCC sequences

CNN Block (Spatial Features):
├── Conv1D(64, kernel=5) + BatchNorm + ReLU + MaxPool
├── Conv1D(128, kernel=5) + BatchNorm + ReLU + MaxPool
└── Conv1D(256, kernel=3) + BatchNorm + ReLU + MaxPool

LSTM Block (Temporal Features):
├── LSTM(128, return_sequences=True) + Dropout(0.3)
└── LSTM(64) + Dropout(0.3)

Classification:
├── Dense(64, relu) + Dropout(0.4)
└── Dense(8, softmax)

Output: 8 emotion probabilities
```

## Dataset

**RAVDESS** - Ryerson Audio-Visual Database of Emotional Speech and Song

- 1440 audio files
- 24 professional actors (12 male, 12 female)
- 8 emotional expressions
- 16-bit, 48kHz audio (resampled to 22050Hz)

### Filename Format

`03-01-06-01-02-01-12.wav`

| Position | Meaning | Values |
|----------|---------|--------|
| 1 | Modality | 03 = audio-only |
| 2 | Vocal | 01 = speech |
| 3 | Emotion | 01-08 |
| 4 | Intensity | 01=normal, 02=strong |
| 5 | Statement | 01="Kids...", 02="Dogs..." |
| 6 | Repetition | 01=1st, 02=2nd |
| 7 | Actor | 01-24 |

## Expected Results

- **Training time**: ~10-15 minutes on M4 Pro (Metal)
- **Expected accuracy**: 70-85%
- **Model size**: ~5-10 MB

## Output Files

After training:
- `models/emotion_cnn_lstm_<timestamp>_best.keras` - Best model
- `models/emotion_cnn_lstm_<timestamp>_final.keras` - Final model
- `outputs/confusion_matrix.png` - Confusion matrix visualization
- `outputs/training_history.png` - Training curves
- `outputs/per_class_metrics.png` - Per-class performance

## Citation

> Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

## License

This project is for educational purposes. The RAVDESS dataset is licensed under Creative Commons BY-NC-SA 4.0.
