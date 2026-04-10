"""Feature extraction tests for robustness feature bundle."""

from __future__ import annotations

import numpy as np

from src.config import AugmentationConfig, FeatureConfig, MAX_LEN
from src.feature_extraction import extract_features


def test_feature_bundle_shape_and_nans(mini_dataset_paths):
    # Pick one sample from generated RAVDESS mini set.
    sample = (
        f"{mini_dataset_paths['ravdess']}/Actor_01/"
        "03-01-03-01-01-01-01.wav"
    )
    features = extract_features(
        sample,
        feature_config=FeatureConfig(
            include_mfcc=True,
            include_delta=True,
            include_delta2=True,
            include_logmel=True,
            include_zcr=True,
            normalize_per_sample=True,
        ),
        augmentation_config=AugmentationConfig(enabled=True),
        training=True,
        rng=np.random.default_rng(123),
    )

    assert features.ndim == 2
    assert features.shape[0] == MAX_LEN
    assert features.shape[1] > 40
    assert np.isfinite(features).all()
