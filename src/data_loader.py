"""Data loading and preprocessing for legacy and multi-dataset SER pipelines."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from src.config import (
    AugmentationConfig,
    FeatureConfig,
    NUM_CLASSES,
    RANDOM_SEED,
    SplitConfig,
)
from src.datasets import load_unified_metadata, metadata_summary
from src.feature_extraction import extract_features
from src.splits import SplitBundle, build_protocol_splits


# ----------------------------
# Legacy RAVDESS helpers
# ----------------------------

def parse_filename(filename: str) -> Dict[str, str]:
    """Parse a RAVDESS filename into metadata components."""
    parts = filename.replace(".wav", "").split("-")
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor": parts[6],
    }


def is_valid_file(filename: str) -> bool:
    """Check if a RAVDESS file is audio-only speech."""
    if not filename.endswith(".wav"):
        return False
    metadata = parse_filename(filename)
    return metadata["modality"] == "03" and metadata["vocal_channel"] == "01"


def get_emotion_label(filename: str) -> int:
    """Extract 0-indexed emotion label from RAVDESS filename."""
    metadata = parse_filename(filename)
    emotion_code = metadata["emotion"]
    return int(emotion_code) - 1


# ----------------------------
# Unified pipeline helpers
# ----------------------------

def _build_tensor_from_metadata(
    metadata: pd.DataFrame,
    feature_config: FeatureConfig,
    augmentation_config: Optional[AugmentationConfig],
    training: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert metadata records to feature tensor and labels."""
    if metadata.empty:
        raise ValueError("Cannot build tensors from empty metadata.")

    rng = np.random.default_rng(seed)
    features: List[np.ndarray] = []
    labels: List[int] = []
    dataset_ids: List[str] = []

    iterator = metadata.itertuples(index=False)
    for row in tqdm(iterator, total=len(metadata), desc="Extracting features", leave=False):
        try:
            matrix = extract_features(
                row.path,
                feature_config=feature_config,
                augmentation_config=augmentation_config,
                training=training,
                rng=rng,
            )
            features.append(matrix)
            labels.append(int(row.emotion_id))
            dataset_ids.append(str(row.dataset_id))
        except Exception as exc:
            # Skip corrupt files while preserving pipeline progress.
            print(f"Warning: skipping {row.path} due to error: {exc}")

    if not features:
        raise ValueError("Feature extraction produced no samples.")

    X = np.stack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    d = np.array(dataset_ids, dtype=object)
    return X, y, d


def _prepare_split_arrays(
    split: SplitBundle,
    feature_config: FeatureConfig,
    augmentation_config: Optional[AugmentationConfig],
    seed: int,
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """Build train/val/test tensors from split metadata."""
    X_train, y_train_raw, _ = _build_tensor_from_metadata(
        split.train,
        feature_config=feature_config,
        augmentation_config=augmentation_config,
        training=True,
        seed=seed,
    )
    X_val, y_val_raw, _ = _build_tensor_from_metadata(
        split.val,
        feature_config=feature_config,
        augmentation_config=None,
        training=False,
        seed=seed + 1,
    )
    X_test, y_test_raw, test_dataset_ids = _build_tensor_from_metadata(
        split.test,
        feature_config=feature_config,
        augmentation_config=None,
        training=False,
        seed=seed + 2,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "y_test_raw": y_test_raw,
        "y_train": to_categorical(y_train_raw, NUM_CLASSES),
        "y_val": to_categorical(y_val_raw, NUM_CLASSES),
        "y_test": to_categorical(y_test_raw, NUM_CLASSES),
        "test_dataset_ids": test_dataset_ids,
        "train_metadata": split.train,
        "val_metadata": split.val,
        "test_metadata": split.test,
        "protocol": split.protocol_name,
    }


def prepare_multidataset_data(
    datasets: Sequence[str],
    protocol: str,
    feature_config: Optional[FeatureConfig] = None,
    augmentation_config: Optional[AugmentationConfig] = None,
    split_config: Optional[SplitConfig] = None,
    dataset_paths: Optional[Dict[str, str]] = None,
    strict_datasets: bool = False,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, np.ndarray | pd.DataFrame | dict]]:
    """
    Prepare data for one or multiple protocols.

    Returns a dictionary keyed by protocol name:
        {
          "random_stratified": { ...split arrays... },
          "speaker_independent": { ...split arrays... }
        }
    """
    feature_cfg = feature_config or FeatureConfig()
    split_cfg = split_config or SplitConfig()

    metadata = load_unified_metadata(
        datasets=datasets,
        dataset_paths=dataset_paths,
        strict=strict_datasets,
    )
    summary = metadata_summary(metadata)

    split_map = build_protocol_splits(metadata=metadata, protocol=protocol, config=split_cfg)
    result: Dict[str, Dict[str, np.ndarray | pd.DataFrame | dict]] = {}

    for protocol_name, split in split_map.items():
        arrays = _prepare_split_arrays(
            split,
            feature_config=feature_cfg,
            augmentation_config=augmentation_config,
            seed=seed,
        )
        arrays["metadata_summary"] = summary
        result[protocol_name] = arrays

    return result


# ----------------------------
# Backward-compatible wrappers
# ----------------------------

def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Legacy loader: RAVDESS-only with MFCC features."""
    records = load_unified_metadata(["ravdess"], strict=True)

    legacy_feature_cfg = FeatureConfig(
        include_delta=False,
        include_delta2=False,
        include_logmel=False,
        include_zcr=False,
        normalize_per_sample=False,
    )

    X, y, _ = _build_tensor_from_metadata(
        records,
        feature_config=legacy_feature_cfg,
        augmentation_config=None,
        training=False,
        seed=RANDOM_SEED,
    )
    file_paths = records["path"].tolist()

    print(f"Loaded {len(X)} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y, minlength=NUM_CLASSES)}")
    return X, y, file_paths


def split_data(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy split wrapper kept for compatibility with old notebooks/scripts."""
    from sklearn.model_selection import train_test_split
    from src.config import TEST_RATIO, TRAIN_RATIO, VAL_RATIO

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy API for existing code paths (RAVDESS + random split)."""
    prepared = prepare_multidataset_data(
        datasets=["ravdess"],
        protocol="random",
        feature_config=FeatureConfig(
            include_delta=False,
            include_delta2=False,
            include_logmel=False,
            include_zcr=False,
            normalize_per_sample=False,
        ),
        augmentation_config=None,
        split_config=SplitConfig(),
        strict_datasets=True,
        seed=RANDOM_SEED,
    )["random_stratified"]

    print(f"Train set: {len(prepared['X_train'])} samples")
    print(f"Validation set: {len(prepared['X_val'])} samples")
    print(f"Test set: {len(prepared['X_test'])} samples")

    return (
        prepared["X_train"],
        prepared["X_val"],
        prepared["X_test"],
        prepared["y_train"],
        prepared["y_val"],
        prepared["y_test"],
    )


def get_emotion_name(label: int) -> str:
    """Convert numeric label to canonical emotion name."""
    from src.config import CANONICAL_EMOTIONS

    if label < 0 or label >= len(CANONICAL_EMOTIONS):
        return "unknown"
    return CANONICAL_EMOTIONS[label]
