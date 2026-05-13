# PPT Generation Prompt — Speech Emotion Recognition
## For: Gamma AI (or any AI presentation tool)

---

Create a COMPLETE professional IEEE-style project presentation (PPT content) for my final year mini/major project titled:

# "Speech Emotion Recognition Using CNN–LSTM and Attention Mechanism for Mental Health Monitoring"

---

## IMPORTANT REQUIREMENTS

- Generate content slide-by-slide
- Include PROFESSIONAL architecture diagrams using Mermaid or ASCII diagrams
- Include workflow figures
- Include model architecture diagrams
- Include frontend/backend architecture diagrams
- Include tables wherever needed
- Include technical explanation suitable for viva and project evaluation
- Include AI/mental health relevance
- Use formal technical language
- Explain all models clearly
- Include input/output dimensions wherever possible
- Make the presentation industry-level and research-oriented
- DO NOT generate literature survey
- DO NOT mention fake data or dummy outputs
- Use ONLY the real benchmark numbers provided below
- Assume real emotion labels and real audio datasets are used
- Include deployment-ready system explanation
- Include real-time prediction workflow explanation

---

## PROJECT OVERVIEW

This project focuses on automatic Speech Emotion Recognition (SER) using Deep Learning for mental health monitoring and emotional analysis.

The system detects human emotions from speech/audio signals using:

1. Deep Learning Models (CNN-LSTM + Attention)
2. Audio Feature Extraction (MFCC, Log-Mel, ZCR)
3. Real-Time WebSocket Streaming
4. FastAPI Backend
5. React + Vite Frontend
6. Ensemble Prediction (Weighted Average)
7. Runtime Model Preset Switching

The project supports:

- Offline audio file prediction (upload WAV/MP3/FLAC/OGG/M4A)
- Real-time microphone emotion recognition via WebSocket
- Emotion history tracking (MongoDB)
- Live emotion streaming with rolling result display
- User authentication (JWT)
- Analytics and weekly insights dashboard
- Runtime model preset selection (3 presets, no restart required)

Final output per prediction:

- Predicted Emotion Label
- Confidence Score (%)
- Emotion Probability Distribution (all 8 emotions)
- Emotion-based Wellness Suggestions

Supported emotions (8-class taxonomy):

| ID | Emotion   |
|----|-----------|
| 0  | Neutral   |
| 1  | Calm      |
| 2  | Happy     |
| 3  | Sad       |
| 4  | Angry     |
| 5  | Fearful   |
| 6  | Disgust   |
| 7  | Surprised |

---

## DATASET DETAILS

Four benchmark emotional speech datasets used in training:

| Dataset  | Speakers        | Files (~) | Emotions | Notes                          |
|----------|-----------------|-----------|----------|--------------------------------|
| RAVDESS  | 24 (12M + 12F)  | 1,440     | 8        | Speech-only, WAV               |
| CREMA-D  | 91 (48M + 43F)  | 7,442     | 6        | No calm/surprised              |
| TESS     | 2 female        | 2,800     | 7        | Studio quality, folder-based   |
| SAVEE    | 4 male          | 480       | 7        | Naturalistic speech            |

All datasets normalized to the canonical 8-emotion taxonomy above.

Dataset ingestion pipeline:

- Audio file discovery from dataset folders
- Metadata extraction per file
- Emotion label normalization across all datasets
- Speaker ID extraction
- Audio validation (sample rate, duration)
- Unified metadata DataFrame generation

Unified metadata schema:

| Column      | Type   | Description                        |
|-------------|--------|------------------------------------|
| dataset_id  | string | ravdess / crema_d / tess / savee   |
| speaker_id  | string | {dataset}:{speaker_raw}            |
| emotion_id  | int    | 0–7 (canonical taxonomy)           |
| sr          | int    | Sample rate (Hz)                   |
| duration    | float  | Audio length (seconds)             |
| path        | string | Absolute path to WAV file          |

Two evaluation protocols:

- **Random Stratified Split** — 70/15/15 train/val/test, stratified by emotion. Same speakers may appear in train and test. Optimistic estimate.
- **Speaker Independent Split** — Held-out speakers never seen during training. Harder, more realistic for deployment.

---

## AUDIO PROCESSING PIPELINE

```
Input: Speech audio (.wav / .mp3 / .flac / .ogg / .m4a / .webm)
          |
          v
    [Audio Loading — librosa]
    Resample to 22050 Hz
    Convert to mono
          |
          v
    [Padding / Truncation]
    Fixed length: 3.0 seconds
    (66,150 samples @ 22050 Hz)
          |
          v
    [Feature Extraction]
    MFCC (40 coefficients)
    Delta MFCC (1st order)
    Delta-Delta MFCC (2nd order)
    Log-Mel Spectrogram (64 mel bins)
    Zero Crossing Rate (ZCR)
          |
          v
    [Per-sample Z-score Normalization]
          |
          v
    Fixed-length Feature Matrix → Model Input
```

Feature extraction configuration:

| Parameter   | Value      |
|-------------|------------|
| Sample Rate | 22,050 Hz  |
| Duration    | 3.0 s      |
| N_MFCC      | 40         |
| N_MELS      | 64         |
| HOP_LENGTH  | 512        |
| N_FFT       | 2048       |
| MAX_LEN     | 130 frames |

Feature vector total width: 40 (MFCC only — lightweight model) or full bundle (attention model).

---

## AUDIO AUGMENTATION TECHNIQUES

Applied during training only to improve generalization:

| Technique          | Probability | Description                                  |
|--------------------|-------------|----------------------------------------------|
| Noise Injection    | 35%         | Adds Gaussian noise (scale=0.005)            |
| Time Shifting      | 25%         | Shifts audio ±0.25 s                         |
| Speed Perturbation | 20%         | Rate between 0.90× and 1.10×                |
| Pitch Shift        | 20%         | ±1.5 semitones                               |
| SpecAugment        | 25%         | Time + frequency masking on spectrogram      |

SpecAugment parameters: max 2 time masks (size 12), max 2 frequency masks (size 12).

---

## MODEL ARCHITECTURE

### 1. Baseline CNN-LSTM Model

```
Input Feature Matrix [batch, timesteps, features]
        |
        v
  Conv1D(64, kernel=3, ReLU)
        |
  BatchNormalization
        |
  MaxPooling1D
        |
  Conv1D(128, kernel=3, ReLU)
        |
  BatchNormalization
        |
  MaxPooling1D
        |
  Conv1D(256, kernel=3, ReLU)
        |
  BatchNormalization
        |
  LSTM(128, return_sequences=True)
        |
  LSTM(64)
        |
  Dense(128, ReLU) + Dropout
        |
  Dense(8, Softmax)
        |
  Output: [batch, 8] — emotion probabilities
```

Purpose:
- CNN layers extract local temporal patterns from feature sequences
- LSTM layers model long-range temporal dependencies
- Suitable for baseline emotion classification

---

### 2. Attention-Based Model (Primary / Production Model)

```
Input Feature Matrix [batch, timesteps, features]
        |
        v
  Conv1D Block (local feature extraction)
        |
  Bidirectional LSTM (forward + backward context)
        |
  Multi-Head Self-Attention
  (learns which time-steps matter most)
        |
  Residual Connection + Layer Normalization
        |
  Global Average Pooling
        |
  Dense(128, ReLU) + Dropout
        |
  Dense(8, Softmax)
        |
  Output: [batch, 8] — emotion probabilities
```

Purpose:
- Bidirectional LSTM captures both past and future audio context
- Self-attention identifies emotionally salient speech segments (stressed syllables, pitch peaks)
- Residual connections prevent gradient vanishing in deep layers
- Best single-model accuracy in the project

---

### 3. Lightweight Model

```
Input Feature Matrix [batch, timesteps, 40]
        |
        v
  SeparableConv1D(64) — depthwise separable convolution
        |
  SeparableConv1D(128)
        |
  Lightweight LSTM(64)
        |
  Dense(8, Softmax)
        |
  Output: [batch, 8] — emotion probabilities
```

Purpose:
- Depthwise separable convolutions reduce parameter count vs standard Conv1D
- Designed for fast inference in real-time streaming context
- Used as ensemble partner (35% weight) alongside attention model (65% weight)

---

## ENSEMBLE PREDICTION SYSTEM

```
Audio Input
    |
    +------------------+------------------+
    |                                     |
    v                                     v
[Lightweight Model]               [Attention Model]
 (Random Stratified)            (Speaker Independent
  Feature: 40 MFCC               or Random Stratified)
  Full feature bundle
    |                                     |
    v                                     v
 Probabilities[8]               Probabilities[8]
    |                                     |
    +-----------> Weighted Average <------+
                  (35% LW + 65% AT)
                         |
                         v
              Ensemble Probabilities[8]
                         |
                         v
              argmax → Predicted Emotion
              max    → Confidence Score
```

Three runtime-switchable presets (no server restart required):

| Preset      | Lightweight        | Attention           | LW%  | AT%  | Accuracy       |
|-------------|-------------------|---------------------|------|------|----------------|
| Current     | Random Stratified | Speaker Independent | 35%  | 65%  | ~66% RS / ~44% SI |
| Consistent  | Random Stratified | Random Stratified   | 35%  | 65%  | ~66% RS        |
| TESS Best   | —                 | Random Stratified   | 0%   | 100% | 99.4% on TESS  |

---

## TRAINING PIPELINE

```
Raw Datasets (RAVDESS, CREMA-D, TESS, SAVEE)
        |
        v
  Dataset Scanner → Unified Metadata DataFrame
        |
        v
  Data Splitter
  (Random Stratified OR Speaker Independent)
        |
        v
  Feature Extractor (MFCC + Delta + LogMel + ZCR)
        |
        v
  Augmentation Pipeline (training split only)
        |
        v
  Tensor Generator → (X_train, y_train, X_val, y_val, X_test, y_test)
        |
        v
  Model Training (Adam, lr=1e-3, batch=32, epochs=100)
  + EarlyStopping (patience=15)
  + ReduceLROnPlateau (factor=0.5, patience=8)
  + ModelCheckpoint (save best val_accuracy)
  + TensorBoard logging
        |
        v
  Best Model Saved → .keras checkpoint
        |
        v
  Evaluation on Test Set
        |
        v
  Benchmark Report (JSON + Markdown)
```

Training configuration:

| Parameter     | Value                         |
|---------------|-------------------------------|
| Batch Size    | 32                            |
| Max Epochs    | 100 (early stopping active)   |
| Learning Rate | 1e-3 (Adam)                   |
| Loss          | Categorical Focal Loss        |
| Class Weights | Yes (imbalance correction)    |
| Random Seed   | 42                            |
| Train/Val/Test| 70 / 15 / 15                  |

---

## BACKEND ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Auth Module  │  │  Prediction  │  │  Analytics   │  │
│  │ JWT + bcrypt │  │   Module     │  │   Module     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│  ┌──────▼─────────────────▼──────────────────▼───────┐  │
│  │               Route Layer                         │  │
│  │  POST /auth/register    POST /auth/login           │  │
│  │  GET  /auth/me          POST /predict-emotion      │  │
│  │  GET  /history          GET  /weekly-analysis      │  │
│  │  GET  /config/model     POST /config/model/{name}  │  │
│  │  WS   /ws/emotion-stream                           │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │              Inference Engine                      │  │
│  │   Lightweight Model (35%)  +  Attention Model (65%)│  │
│  │   librosa feature extraction + ensemble averaging  │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │                  MongoDB                           │  │
│  │   emotion_logs collection   users collection       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

Key backend technologies:

| Component      | Technology                         |
|----------------|------------------------------------|
| Framework      | FastAPI (Python)                   |
| Auth           | JWT (python-jose) + bcrypt         |
| ML Runtime     | TensorFlow / Keras                 |
| Audio          | librosa + soundfile + ffmpeg       |
| Database       | MongoDB (motor async driver)       |
| WebSocket      | FastAPI native WebSocket           |
| Server         | Uvicorn (ASGI)                     |

---

## FRONTEND ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│                   React + Vite Frontend                  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                  React Router                    │    │
│  │   /login   /   /analytics  /history  /settings  │    │
│  └───────────────────┬─────────────────────────────┘    │
│                      │                                   │
│  ┌───────────────────▼─────────────────────────────┐    │
│  │                  Pages / Components              │    │
│  │                                                  │    │
│  │  Dashboard         Analytics       History       │    │
│  │  ├─ AudioRecorder  ├─ Charts       ├─ Table      │    │
│  │  ├─ LiveStream     └─ Insights     └─ Filter     │    │
│  │  ├─ EmotionCard                                  │    │
│  │  └─ SuggestionCard  Settings                     │    │
│  │                     └─ Model Preset Selector     │    │
│  └───────────────────┬─────────────────────────────┘    │
│                      │                                   │
│  ┌───────────────────▼─────────────────────────────┐    │
│  │               Auth Context (JWT)                 │    │
│  │   token stored in state, passed in headers       │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

Frontend technologies:

| Technology    | Purpose                              |
|---------------|--------------------------------------|
| React + Vite  | UI framework + build tool            |
| React Router  | Client-side routing, protected routes|
| TailwindCSS   | Utility-first styling                |
| Recharts      | Live emotion graph visualization     |
| Lucide Icons  | Icon set                             |
| MediaRecorder API | Browser microphone capture       |
| WebSocket API | Live streaming connection            |
| Custom CSS    | Animated waveform bars               |

---

## REAL-TIME EMOTION STREAMING PIPELINE

```
User Microphone
      |
      v
MediaRecorder API (browser)
Restarts every 3 seconds →
generates complete, self-contained WebM file per chunk
      |
      v
WebSocket (ws://localhost:8000/ws/emotion-stream?token=JWT)
      |
      v
FastAPI WebSocket Handler
      |
      v
ffmpeg: WebM → WAV (22050 Hz, mono)
      |
      v
librosa: feature extraction
(MFCC + Delta + Log-Mel + ZCR)
      |
      v
Lightweight Model → probabilities[8]
Attention Model   → probabilities[8]
      |
      v
Weighted ensemble (35% + 65%)
      |
      v
JSON response → WebSocket → Browser
{
  "emotion": "happy",
  "confidence": 0.82,
  "all_scores": { "neutral":0.05, "happy":0.82, ... },
  "timestamp": "2026-05-13T..."
}
      |
      v
React state update
→ Live emotion badge (LIVE)
→ Readings counter
→ Emotion history pills (last 4)
→ Recharts live graph update
      |
      v
MongoDB: emotion_log inserted per chunk
```

Note: MediaRecorder is restarted every 3 seconds (not `.start(3000)`) to ensure each sent blob has a valid WebM container header, making it independently decodable by ffmpeg on the backend.

---

## AUTHENTICATION FLOW

```
Register:
User fills name/email/password
        |
        v
POST /auth/register
        |
bcrypt hash password
        |
Store in MongoDB users collection
        |
Generate JWT (24h expiry)
        |
Return token to frontend
        |
Store in React AuthContext


Login:
POST /auth/login
        |
Find user by email in MongoDB
        |
bcrypt verify password
        |
Generate JWT
        |
Return token


Protected endpoints:
All /predict-emotion, /history, /weekly-analysis
require Bearer token in Authorization header.

WebSocket:
Token passed as query param:
ws://...?token=JWT
```

---

## SYSTEM WORKFLOW (END-TO-END)

```
┌──────────────────────────────────────────────────────────────┐
│  User opens browser → React App                              │
│         ↓                                                    │
│  Login / Register → JWT token obtained                       │
│         ↓                                                    │
│  Dashboard loads                                             │
│         ↓                                                    │
│  ┌──────────────┬──────────────────┬──────────────────┐      │
│  │ Upload Audio │  Record Audio    │  Live Stream     │      │
│  │   (.wav etc) │  (MediaRecorder) │  (WebSocket)     │      │
│  └──────┬───────┴────────┬─────────┴──────────┬───────┘      │
│         └────────────────┴───────────────────-┘              │
│                          ↓                                   │
│         POST /predict-emotion OR WebSocket chunk             │
│                          ↓                                   │
│              FastAPI Backend                                 │
│                          ↓                                   │
│         ffmpeg decode → librosa features                     │
│                          ↓                                   │
│         Ensemble model prediction                            │
│                          ↓                                   │
│         Save to MongoDB emotion_logs                         │
│                          ↓                                   │
│         JSON response → Frontend                             │
│                          ↓                                   │
│  ┌───────────────────────────────────────────────────┐       │
│  │ EmotionCard: label, emoji, confidence, breakdown  │       │
│  │ SuggestionCard: personalized wellness tips        │       │
│  │ Live graph: rolling emotion timeline              │       │
│  └───────────────────────────────────────────────────┘       │
│         ↓                                                    │
│  History page → all past emotion_logs from MongoDB           │
│  Analytics page → weekly trend, dominant emotion             │
│  Settings page → switch model preset (3 options)             │
└──────────────────────────────────────────────────────────────┘
```

---

## EVALUATION PROTOCOLS

Two protocols evaluated independently:

**Protocol 1: Random Stratified Split**
- 70% train / 15% val / 15% test
- Emotion-stratified — all classes proportionally represented
- Same speakers may appear across splits
- Represents upper-bound accuracy estimate

**Protocol 2: Speaker Independent Split**
- Held-out speakers never seen during training
- Tests true generalization to new voices
- Harder, more realistic for real-world deployment
- Industry-standard evaluation for SER

---

## REAL BENCHMARK RESULTS

All numbers from actual training and evaluation runs. Do NOT modify these values.

### Overall Metrics

| Protocol            | Accuracy | Macro F1 | UAR   | Weighted F1 |
|---------------------|----------|----------|-------|-------------|
| Random Stratified   | 66.6%    | 68.3%    | 68.9% | 66.4%       |
| Speaker Independent | 44.5%    | 44.2%    | 46.0% | 42.9%       |

### Per-Dataset Metrics (Random Stratified)

| Dataset  | Samples | Accuracy | Macro F1 | UAR   |
|----------|---------|----------|----------|-------|
| TESS     | 842     | 99.4%    | 99.4%    | 99.3% |
| CREMA-D  | 1,106   | 49.0%    | 47.6%    | 49.9% |
| SAVEE    | 81      | 39.5%    | 33.1%    | 34.5% |
| RAVDESS  | 216     | 39.4%    | 37.8%    | 42.2% |

### Per-Dataset Metrics (Speaker Independent)

| Dataset  | Samples | Accuracy | Macro F1 | UAR   |
|----------|---------|----------|----------|-------|
| TESS     | 2       | 100.0%   | 100.0%   | 100.0%|
| CREMA-D  | 1,224   | 46.4%    | 37.8%    | 39.5% |
| SAVEE    | 120     | 35.0%    | 31.4%    | 35.0% |
| RAVDESS  | 120     | 33.3%    | 27.4%    | 31.3% |

### Paper Comparison

| Paper               | Reported Acc | Our Best (RS) | We Beat? |
|---------------------|-------------|---------------|----------|
| Ouyang (2025)       | 61.3%       | 66.6%         | ✅ Yes   |
| Salian et al. (2021)| 89.3%       | 66.6%         | ❌ No    |
| Ullah et al. (2023) | 82.3%       | 66.6%         | ❌ No    |
| Bhanbhro et al.(2025)| 96.0%      | 66.6%         | ❌ No    |

Note: The high TESS accuracy (99.4%) reflects the dataset's controlled studio recording conditions with only 2 speakers. Real-world accuracy on diverse speakers (CREMA-D: 49%, RAVDESS: 39.4%) reflects the difficulty of cross-speaker generalization.

---

## SLIDES TO GENERATE

1. Title Slide
2. Abstract
3. Introduction
4. Problem Statement
5. Objectives
6. Existing System & Drawbacks
7. Proposed System Overview
8. Dataset Description (table + dataset comparison)
9. Audio Processing Pipeline (diagram)
10. Feature Extraction (diagram + table)
11. Audio Augmentation Techniques (table)
12. CNN-LSTM Architecture (diagram)
13. Attention-Based Architecture (diagram)
14. Lightweight Model Architecture (diagram)
15. Ensemble Learning Architecture (diagram)
16. Model Preset Selector (runtime switching, 3 presets)
17. Complete System Workflow (end-to-end diagram)
18. Backend Architecture (diagram)
19. Frontend Architecture (diagram)
20. FastAPI + React Communication Flow
21. WebSocket Streaming Pipeline (detailed diagram)
22. Authentication Flow (diagram)
23. MongoDB Data Flow
24. Training Pipeline (diagram)
25. Evaluation Protocols (Random Stratified vs Speaker Independent)
26. Performance Metrics — Definitions (Accuracy, F1, UAR, Macro-F1)
27. Results — Overall Table
28. Results — Per-Dataset Table
29. Results — Paper Comparison Table
30. Confusion Matrix Explanation
31. Emotion-wise Prediction Analysis
32. Mental Health Monitoring Relevance
33. Advantages of Proposed System
34. Applications
35. Future Scope
36. Conclusion

---

## DIAGRAMS TO GENERATE

1. Complete System Architecture
2. Audio Processing Workflow
3. Feature Extraction Pipeline
4. CNN-LSTM Architecture
5. Attention Model Architecture (with multi-head attention block)
6. Lightweight Model Architecture
7. Ensemble Prediction Pipeline
8. Backend Architecture
9. Frontend Architecture
10. FastAPI + React Communication
11. WebSocket Streaming Flow (including MediaRecorder restart logic + ffmpeg step)
12. Authentication Flow
13. MongoDB Data Flow
14. Training Pipeline
15. Evaluation Workflow (dual protocol)
16. Real-Time Prediction Workflow
17. Data Flow Diagram (user → frontend → backend → model → DB → response)
18. Deployment Architecture

---

## NOTES FOR PRESENTER

- The speaker-independent result (44.5%) is the honest real-world number. The random-stratified result (66.6%) is the upper-bound. Both are valid and should be presented together.
- TESS achieves 99.4% because it has only 2 speakers in a clean studio — the model effectively memorizes those voices in random-split mode.
- The ensemble (LW 35% + Attention 65%) is justified: attention model captures temporal context better; lightweight provides fast supplementary signal.
- The MediaRecorder restart approach (every 3s) is a deliberate engineering decision — WebM format requires the EBML container header for independent decoding. Simple `start(3000)` produces headerless fragments.
- The system is deployment-ready: JWT auth, MongoDB persistence, async FastAPI, React frontend, configurable model presets.
