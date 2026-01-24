"""Configuration settings for Speech Emotion Recognition."""

import os

# Paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, "archive")
MODEL_PATH = os.path.join(BASE_PATH, "models")
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")

# Audio parameters (RAVDESS is 48kHz, resample to 22050Hz)
SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048

# Computed parameters
MAX_LEN = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1  # ~130 frames

# Model parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Emotions mapping (position 3 in RAVDESS filename)
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

NUM_CLASSES = len(EMOTIONS)

# Random seed for reproducibility
RANDOM_SEED = 42
