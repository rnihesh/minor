"""Feature extraction and augmentation utilities for SER."""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np

from src.config import (
    AugmentationConfig,
    DURATION,
    FeatureConfig,
    HOP_LENGTH,
    MAX_LEN,
    N_FFT,
    N_MFCC,
    SAMPLE_RATE,
)


def _target_num_samples(sample_rate: int, duration: float) -> int:
    return int(sample_rate * duration)


def _target_num_frames(sample_rate: int, duration: float, hop_length: int) -> int:
    return int((sample_rate * duration) / hop_length) + 1


def load_audio(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
    duration: float = DURATION,
) -> np.ndarray:
    """Load audio, resample, and pad/truncate to a fixed duration."""
    signal, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    target_len = _target_num_samples(sample_rate, duration)

    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)), mode="constant")
    elif len(signal) > target_len:
        signal = signal[:target_len]
    return signal.astype(np.float32)


def extract_mfcc(
    signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """Extract MFCC frames in (time, n_mfcc) layout."""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    return mfcc.T.astype(np.float32)


def pad_or_truncate(features: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """Pad or truncate a time-major feature matrix."""
    if features.shape[0] < max_len:
        pad_width = max_len - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode="constant")
    elif features.shape[0] > max_len:
        features = features[:max_len, :]
    return features.astype(np.float32)


def apply_waveform_augmentations(
    signal: np.ndarray,
    sample_rate: int,
    cfg: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply stochastic waveform augmentations."""
    augmented = signal.copy()

    if rng.random() < cfg.noise_prob:
        noise = rng.normal(0.0, cfg.noise_scale, size=augmented.shape).astype(np.float32)
        augmented = augmented + noise

    if rng.random() < cfg.shift_prob:
        max_shift = int(cfg.max_shift_seconds * sample_rate)
        if max_shift > 0:
            shift = int(rng.integers(-max_shift, max_shift + 1))
            augmented = np.roll(augmented, shift)

    if rng.random() < cfg.speed_prob:
        rate = float(rng.uniform(cfg.min_speed_rate, cfg.max_speed_rate))
        stretched = librosa.effects.time_stretch(augmented, rate=rate)
        target_len = len(signal)
        if len(stretched) < target_len:
            stretched = np.pad(stretched, (0, target_len - len(stretched)), mode="constant")
        else:
            stretched = stretched[:target_len]
        augmented = stretched

    if rng.random() < cfg.pitch_prob:
        steps = float(rng.uniform(-cfg.max_pitch_steps, cfg.max_pitch_steps))
        augmented = librosa.effects.pitch_shift(augmented, sr=sample_rate, n_steps=steps)

    return augmented.astype(np.float32)


def apply_specaugment(
    feature_matrix: np.ndarray,
    cfg: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply SpecAugment-style time/frequency masking."""
    if rng.random() >= cfg.specaugment_prob:
        return feature_matrix

    augmented = feature_matrix.copy()
    time_steps, freq_bins = augmented.shape
    fill_value = float(np.mean(augmented))

    for _ in range(cfg.max_time_masks):
        if time_steps <= 1:
            break
        mask_size = int(rng.integers(1, min(cfg.max_time_mask_size, time_steps) + 1))
        start = int(rng.integers(0, max(1, time_steps - mask_size + 1)))
        augmented[start:start + mask_size, :] = fill_value

    for _ in range(cfg.max_freq_masks):
        if freq_bins <= 1:
            break
        mask_size = int(rng.integers(1, min(cfg.max_freq_mask_size, freq_bins) + 1))
        start = int(rng.integers(0, max(1, freq_bins - mask_size + 1)))
        augmented[:, start:start + mask_size] = fill_value

    return augmented.astype(np.float32)


def extract_feature_matrix(signal: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Extract configurable feature bundle and stack into one matrix."""
    parts = []

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        hop_length=cfg.hop_length,
        n_fft=cfg.n_fft,
    ).astype(np.float32)

    if cfg.include_mfcc:
        parts.append(mfcc.T)
    if cfg.include_delta:
        parts.append(librosa.feature.delta(mfcc).T.astype(np.float32))
    if cfg.include_delta2:
        parts.append(librosa.feature.delta(mfcc, order=2).T.astype(np.float32))
    if cfg.include_logmel:
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=cfg.sample_rate,
            n_mels=cfg.n_mels,
            hop_length=cfg.hop_length,
            n_fft=cfg.n_fft,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        parts.append(log_mel.T)
    if cfg.include_zcr:
        zcr = librosa.feature.zero_crossing_rate(
            y=signal,
            hop_length=cfg.hop_length,
            frame_length=cfg.n_fft,
        ).astype(np.float32)
        parts.append(zcr.T)

    if not parts:
        raise ValueError("FeatureConfig has no active features.")

    feature_matrix = np.concatenate(parts, axis=1).astype(np.float32)
    max_len = _target_num_frames(cfg.sample_rate, cfg.duration, cfg.hop_length)
    feature_matrix = pad_or_truncate(feature_matrix, max_len=max_len)

    if cfg.normalize_per_sample:
        mean = np.mean(feature_matrix, axis=0, keepdims=True)
        std = np.std(feature_matrix, axis=0, keepdims=True)
        feature_matrix = (feature_matrix - mean) / np.maximum(std, 1e-6)

    return feature_matrix.astype(np.float32)


def extract_features(
    file_path: str,
    feature_config: Optional[FeatureConfig] = None,
    augmentation_config: Optional[AugmentationConfig] = None,
    training: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Complete feature extraction pipeline.

    Backward-compatible defaults preserve MFCC extraction when called as
    `extract_features(file_path)` from legacy code.
    """
    feature_cfg = feature_config or FeatureConfig(
        n_mels=0,
        include_delta=False,
        include_delta2=False,
        include_logmel=False,
        include_zcr=False,
        normalize_per_sample=False,
    )
    if rng is None:
        rng = np.random.default_rng()

    signal = load_audio(file_path, sample_rate=feature_cfg.sample_rate, duration=feature_cfg.duration)

    if training and augmentation_config and augmentation_config.enabled:
        signal = apply_waveform_augmentations(signal, feature_cfg.sample_rate, augmentation_config, rng)

    matrix = extract_feature_matrix(signal, feature_cfg)

    if training and augmentation_config and augmentation_config.enabled:
        matrix = apply_specaugment(matrix, augmentation_config, rng)
    return matrix.astype(np.float32)
