"""Smoke training tests (1 epoch) for both random and speaker protocols."""

from __future__ import annotations

from src.config import FeatureConfig, SplitConfig
from src.data_loader import prepare_multidataset_data
from src.model import create_model


def test_one_epoch_training_smoke_dual_protocol(mini_dataset_paths):
    prepared = prepare_multidataset_data(
        datasets=["ravdess"],
        protocol="dual",
        feature_config=FeatureConfig(
            include_delta=False,
            include_delta2=False,
            include_logmel=False,
            include_zcr=False,
            normalize_per_sample=False,
        ),
        augmentation_config=None,
        split_config=SplitConfig(),
        dataset_paths={"ravdess": mini_dataset_paths["ravdess"]},
        strict_datasets=True,
    )

    assert {"random_stratified", "speaker_independent"} <= set(prepared.keys())

    for protocol_name, split in prepared.items():
        model = create_model(
            variant="lightweight",
            input_shape=split["X_train"].shape[1:],
            use_focal_loss=False,
        )
        history = model.fit(
            split["X_train"],
            split["y_train"],
            validation_data=(split["X_val"], split["y_val"]),
            epochs=1,
            batch_size=8,
            verbose=0,
        )
        assert "loss" in history.history
        assert len(history.history["loss"]) == 1
