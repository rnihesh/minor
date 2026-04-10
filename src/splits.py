"""Split strategies for random and speaker-independent protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import SplitConfig


@dataclass
class SplitBundle:
    """Train/validation/test split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    protocol_name: str


def _as_frame(metadata: pd.DataFrame, indices) -> pd.DataFrame:
    frame = metadata.iloc[indices].copy().reset_index(drop=True)
    return frame


def split_random_stratified(metadata: pd.DataFrame, config: SplitConfig) -> SplitBundle:
    """Random stratified split, similar to common SER paper setup."""
    indices = metadata.index.to_numpy()
    labels = metadata["emotion_id"].to_numpy()

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_ratio,
        random_state=config.random_seed,
        stratify=labels,
    )

    train_val_labels = metadata.iloc[train_val_idx]["emotion_id"].to_numpy()
    val_ratio = config.val_ratio / (config.train_ratio + config.val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=config.random_seed,
        stratify=train_val_labels,
    )

    return SplitBundle(
        train=_as_frame(metadata, train_idx),
        val=_as_frame(metadata, val_idx),
        test=_as_frame(metadata, test_idx),
        protocol_name="random_stratified",
    )


def split_speaker_independent(metadata: pd.DataFrame, config: SplitConfig) -> SplitBundle:
    """Speaker-independent split with zero speaker overlap."""
    indices = metadata.index.to_numpy()
    groups = metadata["speaker_id"].to_numpy()

    unique_groups = metadata["speaker_id"].nunique()
    if unique_groups < 4:
        raise ValueError("Speaker-independent split requires at least 4 unique speakers.")

    gss_outer = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_ratio,
        random_state=config.random_seed,
    )
    train_val_idx, test_idx = next(gss_outer.split(indices, groups=groups))

    remaining = metadata.iloc[train_val_idx].reset_index()
    inner_indices = remaining.index.to_numpy()
    inner_groups = remaining["speaker_id"].to_numpy()
    val_ratio = config.val_ratio / (config.train_ratio + config.val_ratio)

    gss_inner = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=config.random_seed,
    )
    train_inner_idx, val_inner_idx = next(gss_inner.split(inner_indices, groups=inner_groups))

    train_idx = remaining.iloc[train_inner_idx]["index"].to_numpy()
    val_idx = remaining.iloc[val_inner_idx]["index"].to_numpy()

    split = SplitBundle(
        train=_as_frame(metadata, train_idx),
        val=_as_frame(metadata, val_idx),
        test=_as_frame(metadata, test_idx),
        protocol_name="speaker_independent",
    )
    assert_no_speaker_overlap(split)
    return split


def assert_no_speaker_overlap(split: SplitBundle) -> None:
    """Assert strict speaker separation across split partitions."""
    train_speakers = set(split.train["speaker_id"])
    val_speakers = set(split.val["speaker_id"])
    test_speakers = set(split.test["speaker_id"])

    if train_speakers.intersection(val_speakers):
        raise AssertionError("Speaker overlap detected between train and validation sets.")
    if train_speakers.intersection(test_speakers):
        raise AssertionError("Speaker overlap detected between train and test sets.")
    if val_speakers.intersection(test_speakers):
        raise AssertionError("Speaker overlap detected between validation and test sets.")


def build_protocol_splits(
    metadata: pd.DataFrame,
    protocol: str,
    config: SplitConfig,
) -> Dict[str, SplitBundle]:
    """Build one or more split bundles from a protocol selector."""
    protocol_key = protocol.strip().lower()

    if protocol_key == "random":
        split = split_random_stratified(metadata, config)
        return {split.protocol_name: split}
    if protocol_key == "speaker":
        split = split_speaker_independent(metadata, config)
        return {split.protocol_name: split}
    if protocol_key == "dual":
        random_split = split_random_stratified(metadata, config)
        speaker_split = split_speaker_independent(metadata, config)
        return {
            random_split.protocol_name: random_split,
            speaker_split.protocol_name: speaker_split,
        }

    raise ValueError("Unsupported protocol. Use 'random', 'speaker', or 'dual'.")
