"""Data loading and preprocessing module."""

import os
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from src.config import (
    DATA_PATH, EMOTIONS, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)
from src.feature_extraction import extract_features


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse RAVDESS filename to extract metadata.

    Filename format: 03-01-06-01-02-01-12.wav
    Position meanings:
        1: Modality (03=audio-only)
        2: Vocal channel (01=speech)
        3: Emotion (01-08)
        4: Intensity (01=normal, 02=strong)
        5: Statement
        6: Repetition
        7: Actor

    Args:
        filename: RAVDESS filename

    Returns:
        Dictionary with parsed metadata
    """
    parts = filename.replace('.wav', '').split('-')

    return {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': parts[2],
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6]
    }


def is_valid_file(filename: str) -> bool:
    """
    Check if file is audio-only speech (modality=03, vocal_channel=01).

    Args:
        filename: RAVDESS filename

    Returns:
        True if file is valid for our task
    """
    if not filename.endswith('.wav'):
        return False

    metadata = parse_filename(filename)
    return metadata['modality'] == '03' and metadata['vocal_channel'] == '01'


def get_emotion_label(filename: str) -> int:
    """
    Extract emotion label from filename.

    Args:
        filename: RAVDESS filename

    Returns:
        Emotion label as integer (0-7)
    """
    metadata = parse_filename(filename)
    emotion_code = metadata['emotion']
    # Convert to 0-indexed label
    return int(emotion_code) - 1


def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all valid audio files and extract features.

    Returns:
        X: Features array with shape (n_samples, max_len, n_mfcc)
        y: Labels array with shape (n_samples,)
        file_paths: List of file paths
    """
    features = []
    labels = []
    file_paths = []

    # Get all actor folders
    actor_folders = sorted([
        f for f in os.listdir(DATA_PATH)
        if f.startswith('Actor_') and os.path.isdir(os.path.join(DATA_PATH, f))
    ])

    print(f"Found {len(actor_folders)} actor folders")

    # Process each actor folder
    for actor_folder in tqdm(actor_folders, desc="Loading dataset"):
        actor_path = os.path.join(DATA_PATH, actor_folder)

        for filename in os.listdir(actor_path):
            if not is_valid_file(filename):
                continue

            file_path = os.path.join(actor_path, filename)

            try:
                # Extract features
                mfcc = extract_features(file_path)
                features.append(mfcc)

                # Get label
                label = get_emotion_label(filename)
                labels.append(label)

                # Store file path
                file_paths.append(file_path)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    X = np.array(features)
    y = np.array(labels)

    print(f"Loaded {len(X)} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    return X, y, file_paths


def split_data(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Features array
        y: Labels array

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train+val vs test
    test_size = TEST_RATIO
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Second split: train vs val
    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete data preparation pipeline.

    Returns:
        X_train, X_val, X_test: Feature arrays
        y_train, y_val, y_test: One-hot encoded label arrays
    """
    # Load dataset
    X, y, _ = load_dataset()

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # One-hot encode labels
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_val = to_categorical(y_val, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_emotion_name(label: int) -> str:
    """
    Convert numeric label to emotion name.

    Args:
        label: Emotion label (0-7)

    Returns:
        Emotion name as string
    """
    emotion_code = f"{label + 1:02d}"
    return EMOTIONS.get(emotion_code, "unknown")
