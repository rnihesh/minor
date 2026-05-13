# Architecture Diagrams — Speech Emotion Recognition System

---

## 1. Complete System Architecture

```mermaid
graph TB
    User((User))

    subgraph Frontend["Frontend — React + Vite"]
        Login[Login / Register]
        Dashboard[Dashboard]
        Analytics[Analytics]
        History[History]
        Settings[Settings\nModel Preset Selector]
    end

    subgraph Backend["Backend — FastAPI + Uvicorn"]
        Auth[Auth Module\nJWT + bcrypt]
        Predict[Prediction API\nPOST /predict-emotion]
        WSStream[WebSocket\n/ws/emotion-stream]
        HistoryAPI[History API\nGET /history]
        Weekly[Weekly Analysis\nGET /weekly-analysis]
        ConfigAPI[Config API\nGET/POST /config/model]
    end

    subgraph Inference["Inference Engine"]
        LW[Lightweight Model\nSeparableConv1D + LSTM\n35% weight]
        AT[Attention Model\nBiLSTM + MultiHead Attention\n65% weight]
        Ensemble[Weighted Ensemble\n→ Emotion + Confidence]
    end

    subgraph Storage["MongoDB"]
        Users[(users)]
        Logs[(emotion_logs)]
    end

    User --> Login
    Login -->|JWT token| Dashboard
    Dashboard -->|Upload / Record| Predict
    Dashboard -->|Live mic| WSStream
    Analytics --> HistoryAPI
    Analytics --> Weekly
    Settings --> ConfigAPI

    Predict --> Auth
    WSStream --> Auth
    Auth --> Inference
    Predict --> Inference
    WSStream --> Inference

    LW --> Ensemble
    AT --> Ensemble
    Ensemble --> Logs
    Ensemble -->|JSON response| Dashboard
    Auth --> Users
```

---

## 2. Audio Processing Pipeline

```mermaid
flowchart TD
    A[Raw Audio Input\n.wav / .mp3 / .flac / .ogg / .m4a] --> B[librosa.load\nResample to 22050 Hz\nConvert to mono]
    B --> C{Duration check}
    C -->|< 3s| D[Zero-pad to 66150 samples]
    C -->|> 3s| E[Truncate to 66150 samples]
    C -->|= 3s| F[Pass through]
    D --> G
    E --> G
    F --> G

    G[Feature Extraction] --> H[MFCC\n40 coefficients\nshape: T×40]
    G --> I[Delta MFCC\n1st order\nshape: T×40]
    G --> J[Delta-Delta MFCC\n2nd order\nshape: T×40]
    G --> K[Log-Mel Spectrogram\n64 mel bins\nshape: T×64]
    G --> L[Zero Crossing Rate\nshape: T×1]

    H --> M[Concatenate features\nshape: T×185]
    I --> M
    J --> M
    K --> M
    L --> M

    M --> N[Per-sample Z-score Normalization]
    N --> O[Fixed Feature Matrix\nInput to Model]
```

---

## 3. Feature Extraction Detail

```mermaid
flowchart LR
    Audio["Audio Signal\n66150 samples\n@ 22050 Hz"] --> STFT["STFT\nN_FFT=2048\nHOP=512"]

    STFT --> MEL["Mel Filterbank\n64 filters"]
    STFT --> MFCC_block["DCT → MFCC\n40 coefficients"]

    MEL --> LOGMEL["Log-Mel Spectrogram\nT × 64"]
    MFCC_block --> MFCC["MFCC\nT × 40"]
    MFCC --> DELTA["Delta\nT × 40"]
    DELTA --> DELTA2["Delta-Delta\nT × 40"]

    Audio --> ZCR["Zero Crossing Rate\nT × 1"]

    MFCC --> CAT["Concatenate"]
    DELTA --> CAT
    DELTA2 --> CAT
    LOGMEL --> CAT
    ZCR --> CAT

    CAT --> NORM["Z-score Normalize\nper sample"]
    NORM --> OUT["Feature Matrix\nT × 185"]
```

---

## 4. CNN-LSTM Model Architecture

```mermaid
flowchart TD
    IN["Input\nbatch × T × features"] --> C1

    C1["Conv1D — 64 filters, kernel=3\nReLU activation"] --> BN1["BatchNormalization"]
    BN1 --> MP1["MaxPooling1D"]

    MP1 --> C2["Conv1D — 128 filters, kernel=3\nReLU activation"] --> BN2["BatchNormalization"]
    BN2 --> MP2["MaxPooling1D"]

    MP2 --> C3["Conv1D — 256 filters, kernel=3\nReLU activation"] --> BN3["BatchNormalization"]
    BN3 --> MP3["MaxPooling1D"]

    MP3 --> LSTM1["LSTM — 128 units\nreturn_sequences=True"]
    LSTM1 --> LSTM2["LSTM — 64 units"]
    LSTM2 --> DROP1["Dropout 0.3"]
    DROP1 --> D1["Dense — 128 units, ReLU"]
    D1 --> DROP2["Dropout 0.3"]
    DROP2 --> OUT["Dense — 8 units\nSoftmax\nbatch × 8 probabilities"]
```

---

## 5. Attention-Based Model Architecture

```mermaid
flowchart TD
    IN["Input\nbatch × T × features"] --> CONV["Conv1D Block\n64 → 128 filters\nReLU + BatchNorm"]

    CONV --> BILSTM["Bidirectional LSTM\n128 units each direction\noutput: batch × T × 256"]

    BILSTM --> ATTN["Multi-Head Self-Attention\nQuery, Key, Value from LSTM output\nLearns which time-steps matter most"]

    BILSTM --> RESID_IN
    ATTN --> ADD["Add — Residual Connection"]
    RESID_IN["Identity\nResidual path"] --> ADD

    ADD --> LN["Layer Normalization"]
    LN --> GAP["Global Average Pooling\nbatch × T × 256 → batch × 256"]
    GAP --> D1["Dense — 128 units, ReLU"]
    D1 --> DROP["Dropout 0.4"]
    DROP --> OUT["Dense — 8 units\nSoftmax\nbatch × 8 probabilities"]

    style ATTN fill:#dbeafe,stroke:#2563eb
    style BILSTM fill:#ede9fe,stroke:#7c3aed
```

---

## 6. Lightweight Model Architecture

```mermaid
flowchart TD
    IN["Input\nbatch × T × 40\nMFCC only"] --> SEP1

    SEP1["SeparableConv1D — 64 filters\nDepthwise + Pointwise\nReLU"] --> BN1["BatchNormalization"]
    BN1 --> SEP2["SeparableConv1D — 128 filters\nDepthwise + Pointwise\nReLU"] --> BN2["BatchNormalization"]
    BN2 --> MP["MaxPooling1D"]

    MP --> LSTM["LSTM — 64 units"]
    LSTM --> DROP["Dropout 0.3"]
    DROP --> OUT["Dense — 8 units\nSoftmax\nbatch × 8 probabilities"]

    style SEP1 fill:#dcfce7,stroke:#059669
    style SEP2 fill:#dcfce7,stroke:#059669
```

---

## 7. Ensemble Prediction Pipeline

```mermaid
flowchart TD
    AUDIO["Input Audio\n.wav / .webm"] --> FEAT["Feature Extraction\nlibrosa"]

    FEAT --> LW_FEAT["40 MFCC features\nbatch × T × 40"]
    FEAT --> AT_FEAT["Full feature bundle\nbatch × T × 185"]

    LW_FEAT --> LW_MODEL["Lightweight Model\nSeparableConv1D + LSTM"]
    AT_FEAT --> AT_MODEL["Attention Model\nBiLSTM + MultiHead Attention"]

    LW_MODEL --> LW_PROB["P_lw\n8 probabilities"]
    AT_MODEL --> AT_PROB["P_at\n8 probabilities"]

    LW_PROB --> ENS["Weighted Average\n0.35 × P_lw + 0.65 × P_at"]
    AT_PROB --> ENS

    ENS --> NORM["L1 Normalize"]
    NORM --> ARGMAX["argmax → Predicted Emotion"]
    NORM --> MAX["max → Confidence Score"]
    NORM --> ALL["All 8 scores → Distribution"]

    ARGMAX --> OUT["JSON Response\nemotion, confidence,\nall_scores, suggestions"]
    MAX --> OUT
    ALL --> OUT

    style LW_MODEL fill:#dcfce7,stroke:#059669
    style AT_MODEL fill:#dbeafe,stroke:#2563eb
    style ENS fill:#ede9fe,stroke:#7c3aed
```

---

## 8. WebSocket Real-Time Streaming Pipeline

```mermaid
sequenceDiagram
    participant Mic as Browser Microphone
    participant MR as MediaRecorder
    participant WS_C as WebSocket Client
    participant WS_S as FastAPI WebSocket
    participant FFM as ffmpeg
    participant FE as Feature Extractor
    participant MDL as Ensemble Model
    participant DB as MongoDB
    participant UI as React UI

    Mic->>MR: getUserMedia()
    MR->>MR: start() — record 3s chunk
    MR->>MR: stop() → complete WebM blob
    MR->>WS_C: ondataavailable(blob)
    WS_C->>WS_S: send(blob) [binary WebSocket frame]

    WS_S->>FFM: webm → wav\n22050 Hz mono
    FFM->>FE: wav file
    FE->>FE: MFCC + Delta + LogMel + ZCR
    FE->>MDL: feature matrix
    MDL->>MDL: LW predict + AT predict\nweighted ensemble
    MDL->>DB: insert emotion_log
    MDL->>WS_S: {emotion, confidence, all_scores}
    WS_S->>WS_C: JSON message
    WS_C->>UI: setStreamEmotion(data)
    UI->>UI: update badge + history pills\n+ Recharts graph

    Note over MR: Restarts every 3s<br/>Each blob has valid<br/>WebM EBML header
```

---

## 9. Authentication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as React Frontend
    participant BE as FastAPI Backend
    participant DB as MongoDB

    rect rgb(220, 252, 231)
        Note over U,DB: Registration
        U->>FE: name, email, password
        FE->>BE: POST /auth/register
        BE->>DB: find user by email
        DB-->>BE: not found
        BE->>BE: bcrypt.hashpw(password)
        BE->>DB: insert user document
        BE->>BE: create_access_token(sub, email, name)\nexpiry: 24h
        BE-->>FE: {access_token, name, email}
        FE->>FE: store token in AuthContext
    end

    rect rgb(219, 234, 254)
        Note over U,DB: Login
        U->>FE: email, password
        FE->>BE: POST /auth/login
        BE->>DB: find user by email
        DB-->>BE: user document
        BE->>BE: bcrypt.checkpw(password, hash)
        BE->>BE: create_access_token(...)
        BE-->>FE: {access_token, name, email}
        FE->>FE: store token in AuthContext
    end

    rect rgb(237, 233, 254)
        Note over U,BE: Protected Request
        U->>FE: trigger predict / history
        FE->>BE: request + Authorization: Bearer JWT
        BE->>BE: decode_access_token(JWT)\nverify signature + expiry
        BE-->>FE: 200 OK + data
    end
```

---

## 10. Training Pipeline

```mermaid
flowchart TD
    DS["Raw Datasets\nRAVDESS · CREMA-D · TESS · SAVEE"]
    DS --> SCAN["Dataset Scanner\nWAV discovery + label extraction"]
    SCAN --> META["Unified Metadata DataFrame\ndataset_id, speaker_id, emotion_id,\nsr, duration, path"]

    META --> SPLIT{Split Protocol}

    SPLIT -->|Random Stratified| RS["70/15/15 split\nstratified by emotion\nall speakers mixed"]
    SPLIT -->|Speaker Independent| SI["Hold out N speakers\nfor test set\nnever seen in training"]

    RS --> FEAT
    SI --> FEAT

    FEAT["Feature Extraction\nMFCC + Delta + LogMel + ZCR\nper audio file"]
    FEAT --> AUG["Augmentation\nNoise · Shift · Speed\nPitch · SpecAugment\ntraining split only"]

    AUG --> TENSOR["Tensor Generation\nX_train, y_train\nX_val, y_val\nX_test, y_test"]

    TENSOR --> TRAIN["Model Training\nAdam lr=1e-3, batch=32\nmax 100 epochs"]
    TRAIN --> CB["Callbacks\nEarlyStopping patience=15\nReduceLROnPlateau\nModelCheckpoint\nTensorBoard"]

    CB --> BEST["Best checkpoint saved\n.keras file"]
    BEST --> EVAL["Evaluation on test set\nAccuracy · F1 · UAR · Macro-F1"]
    EVAL --> REPORT["Benchmark Report\nJSON + Markdown\nper-dataset · per-emotion"]
```

---

## 11. MongoDB Data Flow

```mermaid
flowchart LR
    subgraph MongoDB["MongoDB — emotion_db"]
        subgraph USERS["users collection"]
            U_DOC["{ _id, name, email,\npassword_hash, created_at }"]
        end
        subgraph LOGS["emotion_logs collection"]
            L_DOC["{ _id, user_id,\nemotion, confidence,\nall_scores{8},\ntimestamp }"]
        end
    end

    REG["POST /auth/register"] -->|insert| USERS
    LOGIN["POST /auth/login"] -->|find by email| USERS
    PREDICT["POST /predict-emotion"] -->|insert| LOGS
    WS["WebSocket /ws/emotion-stream"] -->|insert per chunk| LOGS
    HISTORY["GET /history"] -->|find by user_id\nlimit 50, sort -timestamp| LOGS
    WEEKLY["GET /weekly-analysis"] -->|aggregate last 7 days\nby user_id| LOGS
```

---

## 12. Evaluation Workflow

```mermaid
flowchart TD
    MODEL["Trained Model\n.keras checkpoint"] --> LOAD["Load with custom objects\ncategorical_focal_loss"]

    LOAD --> RS_TEST["Random Stratified\nTest Set"]
    LOAD --> SI_TEST["Speaker Independent\nTest Set"]

    RS_TEST --> RS_PRED["Predictions\nargmax → emotion label"]
    SI_TEST --> SI_PRED["Predictions\nargmax → emotion label"]

    RS_PRED --> METRICS["Metrics Computation\nAccuracy\nMacro F1\nWeighted F1\nUAR\nPer-class Precision/Recall"]
    SI_PRED --> METRICS

    METRICS --> PERDATASET["Per-Dataset Breakdown\nRAVDESS · CREMA-D\nTESS · SAVEE"]
    METRICS --> PAPER["Paper Comparison\nBenchmark vs literature"]
    METRICS --> REPORT["Benchmark Report\nJSON + Markdown"]

    PERDATASET --> CONFMAT["Confusion Matrix\n8×8"]
    PERDATASET --> PERCLASS["Per-class Bar Charts\nPrecision · Recall · F1"]
```

---

## 13. Model Preset Switching (Runtime)

```mermaid
flowchart TD
    START["Server Startup\nlifespan event"] --> LOAD_ALL

    LOAD_ALL["Load ALL model files\ninto _all_models dict"]
    LOAD_ALL --> M1["ser_lightweight_*_random_stratified\n→ _all_models[lw_rs]"]
    LOAD_ALL --> M2["ser_attention_*_speaker_independent\n→ _all_models[at_si]"]
    LOAD_ALL --> M3["ser_attention_*_random_stratified\n→ _all_models[at_rs]"]

    M1 & M2 & M3 --> READY["Server ready\n_active_preset = 'current'"]

    USER["Settings Page\nUser selects preset"] -->|POST /config/model/consistent| SWITCH
    SWITCH["_active_preset = 'consistent'"] --> NEXT_REQ

    NEXT_REQ["Next prediction request\n_process_audio_file()"]
    NEXT_REQ --> READ["Read PRESETS_active_preset_\nselect lw_file + at_file\nlw_weight + at_weight"]
    READ --> INFER["Run inference\nwith selected models"]

    style SWITCH fill:#ede9fe,stroke:#7c3aed
```

---

## 14. Deployment Architecture

```mermaid
flowchart TB
    subgraph Client["Client — Browser"]
        REACT["React App\nVite dev server :5173"]
    end

    subgraph Server["Local Server / VM"]
        subgraph BackendProc["Backend Process"]
            UVICORN["Uvicorn ASGI\n:8000"]
            FASTAPI["FastAPI App"]
            TF["TensorFlow\nKeras Models × 3"]
            LIBROSA["librosa\naudio feature extraction"]
            FFMPEG["ffmpeg\nwebm → wav conversion"]
        end
        MONGO[("MongoDB\n:27017\nemotion_db")]
    end

    REACT -->|HTTP REST\nBearer JWT| UVICORN
    REACT -->|WebSocket\nws://| UVICORN
    UVICORN --> FASTAPI
    FASTAPI --> TF
    FASTAPI --> LIBROSA
    FASTAPI --> FFMPEG
    FASTAPI -->|motor async| MONGO
```
