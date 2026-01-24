"""Feature extraction module for MFCC computation."""

import numpy as np
import librosa
from src.config import SAMPLE_RATE, DURATION, N_MFCC, HOP_LENGTH, N_FFT, MAX_LEN


def load_audio(file_path: str) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to the audio file

    Returns:
        Audio signal as numpy array
    """
    # Load audio and resample from 48kHz to 22050Hz
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    return signal


def extract_mfcc(signal: np.ndarray) -> np.ndarray:
    """
    Extract MFCC features from audio signal.

    Args:
        signal: Audio signal as numpy array

    Returns:
        MFCC features with shape (n_frames, n_mfcc)
    """
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )

    # Transpose to (n_frames, n_mfcc)
    mfcc = mfcc.T

    return mfcc


def pad_or_truncate(mfcc: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """
    Pad or truncate MFCC to fixed length.

    Args:
        mfcc: MFCC features with shape (n_frames, n_mfcc)
        max_len: Target number of frames

    Returns:
        Padded/truncated MFCC with shape (max_len, n_mfcc)
    """
    if mfcc.shape[0] < max_len:
        # Pad with zeros
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    elif mfcc.shape[0] > max_len:
        # Truncate
        mfcc = mfcc[:max_len, :]

    return mfcc


def extract_features(file_path: str) -> np.ndarray:
    """
    Complete feature extraction pipeline for a single audio file.

    Args:
        file_path: Path to the audio file

    Returns:
        MFCC features with shape (MAX_LEN, N_MFCC)
    """
    # Load audio
    signal = load_audio(file_path)

    # Extract MFCC
    mfcc = extract_mfcc(signal)

    # Pad or truncate to fixed length
    mfcc = pad_or_truncate(mfcc)

    return mfcc
