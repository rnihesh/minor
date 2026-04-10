"""Training pipeline for robustness-first speech emotion recognition."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.config import (
    AugmentationConfig,
    BATCH_SIZE,
    EPOCHS,
    FeatureConfig,
    MODEL_PATH,
    OUTPUT_PATH,
    RUNS_PATH,
    SplitConfig,
    TrainingConfig,
    ensure_directories,
)
from src.data_loader import prepare_multidataset_data
from src.evaluate import evaluate_by_dataset, evaluate_model, plot_training_history
from src.model import create_model, get_model_summary


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_callbacks(model_name: str) -> list:
    """Create standard callbacks used in all experiments."""
    ensure_directories()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_PATH, f"{model_name}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join(OUTPUT_PATH, "logs", model_name),
            histogram_freq=0,
        ),
    ]
    return callbacks


def _feature_config_from_bundle(bundle: str) -> FeatureConfig:
    key = bundle.strip().lower()
    if key in {"mfcc", "legacy"}:
        return FeatureConfig(
            include_delta=False,
            include_delta2=False,
            include_logmel=False,
            include_zcr=False,
            normalize_per_sample=False,
        )
    if key in {"robust", "full", "all"}:
        return FeatureConfig(
            include_mfcc=True,
            include_delta=True,
            include_delta2=True,
            include_logmel=True,
            include_zcr=True,
            normalize_per_sample=True,
        )
    raise ValueError("Unsupported feature bundle. Use 'mfcc' or 'robust'.")


def _compute_class_weight_map(y_train_raw: np.ndarray) -> Optional[Dict[int, float]]:
    classes = np.unique(y_train_raw)
    if len(classes) < 2:
        return None

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_raw,
    )
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def _save_run_summary(summary: dict, run_id: str) -> Path:
    ensure_directories()
    output_path = RUNS_PATH / f"{run_id}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def train_model(
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    model_name: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
    protocol: str = "random",
    model_variant: str = "baseline",
    feature_bundle: str = "robust",
    use_focal_loss: bool = False,
    use_augmentation: bool = True,
    class_weighting: bool = True,
    dataset_paths: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Train SER model(s) for one or more protocols.

    Returns:
        Dictionary keyed by protocol name with training/evaluation artifacts.
    """
    ensure_directories()
    set_global_seed(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name is None:
        model_name = f"ser_{model_variant}_{timestamp}"

    selected_datasets = list(datasets) if datasets else ["ravdess"]
    feature_config = _feature_config_from_bundle(feature_bundle)
    augmentation_config = AugmentationConfig(enabled=use_augmentation)
    split_config = SplitConfig()
    training_config = TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        use_focal_loss=use_focal_loss,
        class_weighting=class_weighting,
    )

    print("=" * 70)
    print("Robustness-First SER Training")
    print("=" * 70)
    print(f"Datasets: {selected_datasets}")
    print(f"Protocol: {protocol}")
    print(f"Model variant: {model_variant}")
    print(f"Feature bundle: {feature_bundle}")

    prepared_by_protocol = prepare_multidataset_data(
        datasets=selected_datasets,
        protocol=protocol,
        feature_config=feature_config,
        augmentation_config=augmentation_config if use_augmentation else None,
        split_config=split_config,
        dataset_paths=dataset_paths,
        strict_datasets=False,
        seed=training_config.random_seed,
    )

    all_results: Dict[str, dict] = {}

    for protocol_name, prepared in prepared_by_protocol.items():
        protocol_model_name = f"{model_name}_{protocol_name}"
        print("\n" + "-" * 70)
        print(f"Protocol: {protocol_name}")
        print("-" * 70)

        X_train = prepared["X_train"]
        X_val = prepared["X_val"]
        X_test = prepared["X_test"]
        y_train = prepared["y_train"]
        y_val = prepared["y_val"]
        y_test = prepared["y_test"]
        y_train_raw = prepared["y_train_raw"]

        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"X_test shape: {X_test.shape}")

        model = create_model(
            variant=model_variant,
            input_shape=X_train.shape[1:],
            learning_rate=training_config.learning_rate,
            use_focal_loss=training_config.use_focal_loss,
        )
        print(get_model_summary(model))

        callbacks = get_callbacks(protocol_model_name)
        class_weight = (
            _compute_class_weight_map(y_train_raw)
            if training_config.class_weighting
            else None
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=training_config.batch_size,
            epochs=training_config.epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        final_model_path = os.path.join(MODEL_PATH, f"{protocol_model_name}_final.keras")
        best_model_path = os.path.join(MODEL_PATH, f"{protocol_model_name}_best.keras")
        model.save(final_model_path)

        history_plot_path = os.path.join(OUTPUT_PATH, f"{protocol_model_name}_history.png")
        plot_training_history(history, save_path=history_plot_path)

        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            save_plots=True,
            output_prefix=protocol_model_name,
        )
        per_dataset = evaluate_by_dataset(
            metrics["y_true"],
            metrics["y_pred"],
            prepared["test_dataset_ids"],
        )

        all_results[protocol_name] = {
            "model_variant": model_variant,
            "feature_bundle": feature_bundle,
            "model_name": protocol_model_name,
            "best_model_path": best_model_path,
            "final_model_path": final_model_path,
            "history_plot_path": history_plot_path,
            "metrics": {
                key: value
                for key, value in metrics.items()
                if key
                not in {"y_true", "y_pred", "y_pred_proba", "classification_report"}
            },
            "per_dataset_metrics": per_dataset,
            "metadata_summary": prepared["metadata_summary"],
            "epochs_trained": len(history.history.get("loss", [])),
            "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
            "best_val_loss": float(min(history.history.get("val_loss", [float("inf")]))),
        }

    run_id = f"{model_name}"
    run_summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "datasets": selected_datasets,
        "protocol": protocol,
        "model_variant": model_variant,
        "feature_bundle": feature_bundle,
        "use_focal_loss": use_focal_loss,
        "use_augmentation": use_augmentation,
        "class_weighting": class_weighting,
        "results": all_results,
    }
    summary_path = _save_run_summary(run_summary, run_id)
    print(f"\nRun summary saved to: {summary_path}")

    return run_summary


if __name__ == "__main__":
    train_model()
