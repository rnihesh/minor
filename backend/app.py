"""
Speech Emotion Recognition - FastAPI Backend
=============================================
This server loads two trained Keras models and exposes endpoints for
predicting emotion from audio, fetching history from MongoDB, running weekly
analysis, and a WebSocket for real-time streaming.

Added: JWT-based authentication (register / login).
"""

import os
import subprocess
import sys
import tempfile
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so we can import `src.*`
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import motor.motor_asyncio

from auth import hash_password, verify_password, create_access_token, decode_access_token
from src.config import CANONICAL_EMOTIONS, FeatureConfig
from src.feature_extraction import extract_features

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "current": {
        "label": "Current (Mixed Protocol)",
        "description": "Lightweight random-split + Attention speaker-independent",
        "accuracy": "~66% RS / ~44% SI",
        "lw_file": "ser_lightweight_20260425_235238_random_stratified_best.keras",
        "at_file": "ser_attention_20260425_235453_speaker_independent_best.keras",
        "lw_weight": 0.35,
        "at_weight": 0.65,
    },
    "consistent": {
        "label": "Consistent (Both Random Split)",
        "description": "Both models same protocol — more balanced ensemble, ~66% accuracy",
        "accuracy": "~66% RS",
        "lw_file": "ser_lightweight_20260425_235238_random_stratified_best.keras",
        "at_file": "ser_attention_20260426_004328_random_stratified_best.keras",
        "lw_weight": 0.35,
        "at_weight": 0.65,
    },
    "tess": {
        "label": "TESS Best (Attention Only)",
        "description": "Latest attention model only — 99.4% on TESS, best for clean audio",
        "accuracy": "99.4% TESS / ~66% overall",
        "lw_file": None,
        "at_file": "ser_attention_20260426_004328_random_stratified_best.keras",
        "lw_weight": 0.0,
        "at_weight": 1.0,
    },
}

_active_preset: str = "current"
_all_models: dict = {}  # keyed by filename

# ---------------------------------------------------------------------------
# Lifespan — model loading deferred to worker process (post-fork).
# Loading TF at module level + uvicorn reload=True forks after Metal init
# → mutex lock crash on macOS. Lifespan runs inside the worker, never
# in the reloader process.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
        print("[INFO] Metal/GPU disabled for inference worker.")
    except Exception:
        pass
    from tensorflow.keras.models import load_model
    from src.model import categorical_focal_loss
    custom_objects = {"loss_fn": categorical_focal_loss()}

    # Collect unique model files across all presets
    files_to_load = {
        p["lw_file"] for p in PRESETS.values() if p["lw_file"]
    } | {p["at_file"] for p in PRESETS.values()}

    print(f"[INFO] Loading {len(files_to_load)} model file(s)...")
    for fname in files_to_load:
        path = PROJECT_ROOT / "models" / fname
        _all_models[fname] = load_model(path, custom_objects=custom_objects)
        print(f"[INFO]   ✓ {fname}")
    print("[INFO] All models loaded ✓")
    yield

EMOTION_SUGGESTIONS: dict[str, list[str]] = {
    "neutral": ["You seem balanced – keep it up!", "A good time to reflect on your goals"],
    "calm": ["You sound calm and collected 🧘", "This is a great state for creative work"],
    "happy": ["Great to hear you're happy! 🎉", "Share your joy with someone you care about"],
    "sad": ["Try a short breathing exercise 🌬️", "Listen to calming or uplifting music 🎵", "Talk to someone you trust 💬"],
    "angry": ["Take a few slow, deep breaths 🧘", "Step away for a 5-minute cool-down walk", "Write down what's bothering you ✍️"],
    "fearful": ["Ground yourself: name 5 things you can see", "Reach out to a friend or counsellor 🤝"],
    "disgust": ["Take a moment to step back and breathe", "Redirect your focus to something positive"],
    "surprised": ["Take a moment to process this feeling", "Channel your surprise into curiosity 🔍"],
}

EMOTION_EMOJI: dict[str, str] = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}

# ---------------------------------------------------------------------------
# MongoDB setup
# ---------------------------------------------------------------------------
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client.emotion_db
collection = db.emotion_logs
users_collection = db.users

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

# ---------------------------------------------------------------------------
# JWT dependency
# ---------------------------------------------------------------------------
bearer_scheme = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return payload

# ---------------------------------------------------------------------------
# FastAPI app & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Speech Emotion Recognition API (Extended)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _webm_to_wav(webm_path: str) -> str:
    wav_path = webm_path.replace(".webm", ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, "-ar", "22050", "-ac", "1", "-f", "wav", wav_path],
        capture_output=True, check=True
    )
    return wav_path

def _infer_feature_config(model) -> FeatureConfig:
    feature_bins = int(model.input_shape[-1])
    if feature_bins == 40:
        return FeatureConfig(include_delta=False, include_delta2=False, include_logmel=False, include_zcr=False, normalize_per_sample=False)
    return FeatureConfig(include_mfcc=True, include_delta=True, include_delta2=True, include_logmel=True, include_zcr=True, normalize_per_sample=True)

async def _process_audio_file(file_path: str):
    preset = PRESETS[_active_preset]
    at_model = _all_models[preset["at_file"]]
    lw_weight = preset["lw_weight"]
    at_weight = preset["at_weight"]

    at_cfg = _infer_feature_config(at_model)
    at_features = np.expand_dims(extract_features(file_path, feature_config=at_cfg), axis=0)
    at_probs = at_model.predict(at_features, verbose=0)[0]

    if preset["lw_file"] and lw_weight > 0:
        lw_model = _all_models[preset["lw_file"]]
        lw_cfg = _infer_feature_config(lw_model)
        lw_features = np.expand_dims(extract_features(file_path, feature_config=lw_cfg), axis=0)
        lw_probs = lw_model.predict(lw_features, verbose=0)[0]
        ensemble_probs = (lw_weight * lw_probs) + (at_weight * at_probs)
    else:
        lw_probs = at_probs
        ensemble_probs = at_probs.copy()

    ensemble_probs = ensemble_probs / ensemble_probs.sum()
    predicted_idx = int(np.argmax(ensemble_probs))
    predicted_emotion = CANONICAL_EMOTIONS[predicted_idx]
    confidence = float(ensemble_probs[predicted_idx])
    all_scores = {e: round(float(p), 4) for e, p in zip(CANONICAL_EMOTIONS, ensemble_probs)}

    return predicted_emotion, confidence, all_scores, lw_probs, at_probs

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Speech Emotion API with MongoDB running."}


# ---------------------------------------------------------------------------
# Model config endpoints (no auth — settings page only)
# ---------------------------------------------------------------------------
@app.get("/config/model")
async def get_model_config():
    return {
        "active": _active_preset,
        "presets": {
            k: {
                "label": v["label"],
                "description": v["description"],
                "accuracy": v["accuracy"],
                "lw_weight": v["lw_weight"],
                "at_weight": v["at_weight"],
            }
            for k, v in PRESETS.items()
        },
    }

@app.post("/config/model/{preset_name}")
async def set_model_config(preset_name: str):
    global _active_preset
    if preset_name not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Unknown preset '{preset_name}'. Valid: {list(PRESETS)}")
    _active_preset = preset_name
    return {"active": _active_preset, "label": PRESETS[preset_name]["label"]}


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    existing = await users_collection.find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user_doc = {
        "name": body.name.strip(),
        "email": body.email.lower().strip(),
        "password_hash": hash_password(body.password),
        "created_at": datetime.utcnow(),
    }
    result = await users_collection.insert_one(user_doc)
    token = create_access_token({"sub": str(result.inserted_id), "email": body.email, "name": body.name})
    return {"access_token": token, "token_type": "bearer", "name": body.name, "email": body.email}


@app.post("/auth/login")
async def login(body: LoginRequest):
    user = await users_collection.find_one({"email": body.email.lower().strip()})
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": str(user["_id"]), "email": user["email"], "name": user.get("name", "")})
    return {"access_token": token, "token_type": "bearer", "name": user.get("name", ""), "email": user["email"]}


@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {"email": current_user.get("email"), "name": current_user.get("name")}

@app.post("/predict-emotion")
async def predict_emotion(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    file_ext = Path(file.filename or "upload.wav").suffix.lower()
    
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        tmp.write(await file.read())
        tmp.close()
        tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    try:
        emotion, confidence, all_scores, lw_probs, at_probs = await _process_audio_file(tmp_path)

        # Build response
        response_data = {
            "emotion": emotion,
            "emoji": EMOTION_EMOJI.get(emotion, "🎤"),
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "suggestions": EMOTION_SUGGESTIONS.get(emotion, []),
            "models_used": {
                "lightweight": {"predicted": CANONICAL_EMOTIONS[int(np.argmax(lw_probs))], "confidence": round(float(np.max(lw_probs)), 4)},
                "attention": {"predicted": CANONICAL_EMOTIONS[int(np.argmax(at_probs))], "confidence": round(float(np.max(at_probs)), 4)},
            },
        }

        # Store in MongoDB
        doc = {
            "user_id": current_user.get("sub"),
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "timestamp": datetime.utcnow()
        }
        await collection.insert_one(doc)

        return response_data
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

@app.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    """Return last 50 emotion records from MongoDB."""
    cursor = collection.find({"user_id": current_user.get("sub")}).sort("timestamp", -1).limit(50)
    records = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        records.append(doc)
    return records

@app.get("/weekly-analysis")
async def weekly_analysis(current_user: dict = Depends(get_current_user)):
    """Analyze last 7 days of data for trends and suggestions."""
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    three_days_ago = datetime.utcnow() - timedelta(days=3)

    cursor = collection.find({
        "user_id": current_user.get("sub"),
        "timestamp": {"$gte": seven_days_ago}
    })
    docs = await cursor.to_list(length=1000)

    if not docs:
        return {"message": "No data in the last 7 days", "dominant_emotion": None}

    emotion_counts = {}
    first_half_counts = {"sad": 0, "happy": 0}
    second_half_counts = {"sad": 0, "happy": 0}

    for doc in docs:
        emo = doc["emotion"]
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        
        if doc["timestamp"] < three_days_ago:
            if emo in first_half_counts: first_half_counts[emo] += 1
        else:
            if emo in second_half_counts: second_half_counts[emo] += 1

    dominant_emotion = max(emotion_counts, key=emotion_counts.get)

    trend = "stable"
    if second_half_counts["sad"] > first_half_counts["sad"]:
        trend = "worsening"
    elif second_half_counts["happy"] > first_half_counts["happy"]:
        trend = "improving"

    suggestion = "Your emotions seem balanced. Keep it up!"
    if trend == "worsening" or dominant_emotion in ["sad", "fearful", "angry"]:
        suggestion = "You seem to feel low frequently. Consider relaxation or talking to someone."
    elif dominant_emotion in ["happy", "calm"]:
        suggestion = "You're having a great week! Keep doing what you're doing."

    return {
        "dominant_emotion": dominant_emotion,
        "emotion_counts": emotion_counts,
        "trend": trend,
        "suggestion": suggestion
    }

@app.websocket("/ws/emotion-stream")
async def emotion_stream(websocket: WebSocket, token: str = None):
    """Real-time streaming emotion recognition."""
    user_id = None
    if token:
        try:
            payload = decode_access_token(token)
            if payload:
                user_id = payload.get("sub")
        except:
            pass

    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_bytes()
            except (WebSocketDisconnect, RuntimeError):
                break

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
            tmp.write(data)
            tmp.close()
            webm_path = tmp.name
            wav_path = None

            try:
                wav_path = _webm_to_wav(webm_path)
                emotion, confidence, all_scores, _, _ = await _process_audio_file(wav_path)

                timestamp = datetime.utcnow()
                await collection.insert_one({
                    "user_id": user_id,
                    "emotion": emotion,
                    "confidence": round(confidence, 4),
                    "all_scores": all_scores,
                    "timestamp": timestamp
                })

                try:
                    await websocket.send_json({
                        "emotion": emotion,
                        "confidence": round(confidence, 4),
                        "all_scores": all_scores,
                        "timestamp": timestamp.isoformat()
                    })
                except (WebSocketDisconnect, RuntimeError):
                    break
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception as e:
                print(f"[WS processing error] {e}")
                traceback.print_exc()
            finally:
                for p in (webm_path, wav_path):
                    if p:
                        try:
                            os.unlink(p)
                        except OSError:
                            pass

    except Exception:
        pass
    print("[WS] Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
