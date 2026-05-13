# SER Models — Dataset & Architecture Reference

## Models in Use (Production)

| Role | File | Split Protocol |
|------|------|---------------|
| **Lightweight** (35% weight) | `ser_lightweight_20260425_235238_random_stratified_best.keras` | Random stratified |
| **Attention** (65% weight) | `ser_attention_20260425_235453_speaker_independent_best.keras` | Speaker independent |

The backend ensembles both: `final = 0.35 × lightweight + 0.65 × attention`.

---

## Training Datasets

All models trained on **4 combined datasets**. Audio-only, speech channel only.

### 1. RAVDESS
**Ryerson Audio-Visual Database of Emotional Speech and Song**

| Attribute | Value |
|-----------|-------|
| Actors | 24 (12 male, 12 female) |
| Files used | ~1,440 WAV (modality=03, vocal channel=01 only) |
| Sample rate | 48 kHz (downsampled to 22 050 Hz) |
| Emotions | 8: neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Intensity levels | normal, strong |
| Filename format | `03-01-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav` |
| Emotion codes | 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised |

### 2. CREMA-D
**Crowd-sourced Emotional Multimodal Actors Dataset**

| Attribute | Value |
|-----------|-------|
| Actors | 91 (48 male, 43 female, ages 20–74) |
| Files | ~7,442 WAV |
| Emotions | 6: angry, disgust, fearful, happy, neutral, sad *(no calm, no surprised)* |
| Intensity levels | low, medium, high, unspecified |
| Filename format | `[ActorID]_[SentenceKey]_[Emotion]_[Level].wav` |
| Emotion codes | ANG, DIS, FEA, HAP, NEU, SAD |

### 3. TESS
**Toronto Emotional Speech Set**

| Attribute | Value |
|-----------|-------|
| Speakers | 2 female (ages 26 and 64) |
| Files | ~2,800 WAV |
| Emotions | 7: angry, disgust, fearful, happy, neutral, sad, surprised (`ps` = pleasant surprise) |
| Structure | Folder-based — emotion encoded in parent directory name |
| Filename format | `[speaker]_[word]_[emotion].wav` or emotion in folder name |

### 4. SAVEE
**Surrey Audio-Visual Expressed Emotion**

| Attribute | Value |
|-----------|-------|
| Speakers | 4 male (postgraduate students, ages 27–31) |
| Files | ~480 WAV |
| Emotions | 7: angry, disgust, fearful, happy, neutral, sad, surprised |
| Filename format | `[speaker]_[emotioncode][number].wav` |
| Emotion codes | `a`=angry `d`=disgust `f`=fearful `h`=happy `n`=neutral `sa`=sad `su`=surprised |

---

## Canonical Emotion Taxonomy (8 classes)

All datasets are normalized to this label set:

| ID | Emotion | Notes |
|----|---------|-------|
| 0 | neutral | All 4 datasets |
| 1 | calm | RAVDESS only |
| 2 | happy | All 4 datasets |
| 3 | sad | All 4 datasets |
| 4 | angry | All 4 datasets |
| 5 | fearful | All 4 datasets |
| 6 | disgust | All 4 datasets |
| 7 | surprised | RAVDESS, TESS, SAVEE (not CREMA-D) |

---

## Metadata Schema (per audio sample)

```
dataset_id   — "ravdess" | "crema_d" | "tess" | "savee"
speaker_id   — "{dataset_id}:{speaker_raw}"  e.g. "ravdess:01"
emotion_id   — int 0–7 (maps to canonical taxonomy above)
sr           — sample rate in Hz (int)
duration     — audio length in seconds (float)
path         — absolute path to WAV file
```

---

## Feature Extraction

Each audio file → fixed feature vector fed to the model.

| Feature | Config |
|---------|--------|
| MFCC | 40 coefficients |
| Delta MFCC | 1st order |
| Delta-Delta MFCC | 2nd order |
| Log-Mel spectrogram | 64 mel bins |
| Zero-crossing rate | yes |
| Normalization | per-sample z-score |
| Audio length | trimmed/padded to 3.0 s |
| Sample rate | 22 050 Hz |
| Hop length | 512 |
| FFT size | 2048 |

Feature vector total bins: 40 (MFCC only, lightweight) or full bundle (attention).

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Max epochs | 100 (early stopping) |
| Learning rate | 1e-3 |
| Loss | Categorical focal loss |
| Class weighting | yes (imbalance correction) |
| Augmentation | noise, time-shift, speed, pitch, SpecAugment |
| Random seed | 42 |
| Train/Val/Test split | 70 / 15 / 15 |

---

## Split Protocols

### `random_stratified`
Standard 70/15/15 split, stratified by emotion label. Same speakers can appear in train and test. Optimistic estimate of real-world performance.

### `speaker_independent`
Held-out speakers never appear in training. Harder, more realistic for deployment. Model must generalize to voices it has never heard.

---

## Benchmark Results (ser_attention_20260410 — earlier run, same architecture)

| Protocol | Accuracy | Macro-F1 | UAR |
|----------|----------|----------|-----|
| random_stratified | 66.6% | 68.3% | 68.9% |
| speaker_independent | 44.5% | 44.2% | 46.0% |

Per-dataset (random_stratified):

| Dataset | Accuracy |
|---------|----------|
| TESS | 99.4% |
| CREMA-D | 49.0% |
| SAVEE | 39.5% |
| RAVDESS | 39.4% |

TESS is near-perfect (clean studio recordings, 2 speakers). CREMA-D/SAVEE/RAVDESS are harder due to speaker diversity and naturalistic variation.

---

## Model Files — Full Inventory

| File | Type | Protocol | Status |
|------|------|----------|--------|
| `ser_attention_20260426_004328_random_stratified_best.keras` | Attention | random_stratified | Latest random split |
| `ser_attention_20260425_235453_speaker_independent_best.keras` | Attention | speaker_independent | **Production attention model** |
| `ser_attention_20260425_235453_random_stratified_best.keras` | Attention | random_stratified | — |
| `ser_lightweight_20260425_235238_random_stratified_best.keras` | Lightweight | random_stratified | **Production lightweight model** |
| `emotion_cnn_lstm_20260124_111454_best.keras` | Old CNN-LSTM | — | Deprecated — do not use |
