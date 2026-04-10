"""Split protocol tests for speaker overlap guarantees."""

from __future__ import annotations

from src.config import SplitConfig
from src.datasets import load_unified_metadata
from src.splits import assert_no_speaker_overlap, split_speaker_independent


def test_speaker_independent_split_has_no_overlap(mini_dataset_paths):
    metadata = load_unified_metadata(
        datasets=["ravdess"],
        dataset_paths={"ravdess": mini_dataset_paths["ravdess"]},
        strict=True,
    )
    split = split_speaker_independent(metadata, SplitConfig())
    assert_no_speaker_overlap(split)

    train_speakers = set(split.train["speaker_id"])
    val_speakers = set(split.val["speaker_id"])
    test_speakers = set(split.test["speaker_id"])

    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)
