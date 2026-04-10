"""Data integrity checks for unified metadata ingestion."""

from __future__ import annotations

from src.config import EMOTION_TO_ID
from src.datasets import METADATA_COLUMNS, load_unified_metadata


def test_unified_metadata_schema_and_labels(mini_dataset_paths):
    metadata = load_unified_metadata(
        datasets=["ravdess", "crema_d", "tess", "savee"],
        dataset_paths=mini_dataset_paths,
        strict=True,
    )

    assert not metadata.empty
    assert list(metadata.columns) == METADATA_COLUMNS
    assert metadata["path"].map(lambda p: isinstance(p, str)).all()
    assert metadata["emotion_id"].isin(list(range(len(EMOTION_TO_ID)))).all()
    assert metadata["sr"].notna().all()
    assert metadata["duration"].notna().all()
