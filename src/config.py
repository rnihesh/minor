"""Configuration and defaults for speech emotion recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

# Paths
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "archive"  # Backward-compatibility for existing RAVDESS path
DATA_ROOT = BASE_PATH / "data"
MODEL_PATH = BASE_PATH / "models"
OUTPUT_PATH = BASE_PATH / "outputs"
RUNS_PATH = OUTPUT_PATH / "runs"
REPORTS_PATH = OUTPUT_PATH / "reports"

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 3.0  # seconds
N_MFCC = 40
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 2048
MAX_LEN = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Canonical 8-class SER taxonomy (RAVDESS-compatible)
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
CANONICAL_EMOTIONS = [EMOTIONS[f"{idx:02d}"] for idx in range(1, 9)]
EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(CANONICAL_EMOTIONS)}
NUM_CLASSES = len(CANONICAL_EMOTIONS)

SUPPORTED_DATASETS = ("ravdess", "crema_d", "tess", "savee")
DEFAULT_DATASET_PATHS = {
    "ravdess": [DATA_PATH, DATA_ROOT / "ravdess"],
    "crema_d": [DATA_ROOT / "crema_d"],
    "tess": [DATA_ROOT / "tess"],
    "savee": [DATA_ROOT / "savee"],
}


@dataclass(frozen=True)
class FeatureConfig:
    """Feature extraction configuration."""

    sample_rate: int = SAMPLE_RATE
    duration: float = DURATION
    hop_length: int = HOP_LENGTH
    n_fft: int = N_FFT
    n_mfcc: int = N_MFCC
    n_mels: int = N_MELS
    include_mfcc: bool = True
    include_delta: bool = True
    include_delta2: bool = True
    include_logmel: bool = True
    include_zcr: bool = True
    normalize_per_sample: bool = True


@dataclass(frozen=True)
class AugmentationConfig:
    """Waveform and feature augmentation configuration."""

    enabled: bool = True
    noise_prob: float = 0.35
    shift_prob: float = 0.25
    speed_prob: float = 0.20
    pitch_prob: float = 0.20
    specaugment_prob: float = 0.25
    noise_scale: float = 0.005
    max_shift_seconds: float = 0.25
    min_speed_rate: float = 0.90
    max_speed_rate: float = 1.10
    max_pitch_steps: float = 1.5
    max_time_masks: int = 2
    max_freq_masks: int = 2
    max_time_mask_size: int = 12
    max_freq_mask_size: int = 12


@dataclass(frozen=True)
class SplitConfig:
    """Data split configuration."""

    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO
    random_seed: int = RANDOM_SEED


@dataclass(frozen=True)
class TrainingConfig:
    """Model training configuration."""

    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    use_focal_loss: bool = False
    class_weighting: bool = True
    random_seed: int = RANDOM_SEED


def resolve_dataset_root(
    dataset_id: str,
    override_paths: Optional[Dict[str, str | Path]] = None,
) -> Optional[Path]:
    """Resolve the first available path for a dataset."""
    if override_paths and dataset_id in override_paths:
        path = Path(override_paths[dataset_id]).expanduser().resolve()
        return path if path.exists() else None

    candidates: Iterable[Path] = DEFAULT_DATASET_PATHS.get(dataset_id, [])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def ensure_directories() -> None:
    """Create standard output directories if missing."""
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    RUNS_PATH.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
