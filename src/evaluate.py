"""Evaluation and reporting utilities for SER models."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.models import Model, load_model

from src.config import CANONICAL_EMOTIONS, MODEL_PATH, OUTPUT_PATH
from src.data_loader import prepare_data, prepare_multidataset_data


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute weighted and macro classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision_w),
        "recall": float(recall_w),
        "f1_score": float(f1_w),
        "macro_f1": float(f1_macro),
        "uar": float(uar),
    }


def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_plots: bool = True,
    output_prefix: str = "",
) -> dict:
    """Evaluate a model on a test tensor."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    metrics = compute_classification_metrics(y_true, y_pred)

    print("\n" + "=" * 60)
    print("Model Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"UAR:       {metrics['uar']:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CANONICAL_EMOTIONS))),
        target_names=CANONICAL_EMOTIONS,
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)

    if save_plots:
        prefix = f"{output_prefix}_" if output_prefix else ""
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=CANONICAL_EMOTIONS,
            save_path=os.path.join(OUTPUT_PATH, f"{prefix}confusion_matrix.png"),
        )
        plot_per_class_metrics(
            y_true=y_true,
            y_pred=y_pred,
            save_path=os.path.join(OUTPUT_PATH, f"{prefix}per_class_metrics.png"),
        )

    return {
        **metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "classification_report": report,
    }


def evaluate_by_dataset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_ids: Sequence[str],
) -> Dict[str, dict]:
    """Compute per-dataset metrics from test predictions."""
    result: Dict[str, dict] = {}
    ids = np.array(dataset_ids)

    for dataset_id in sorted(set(ids.tolist())):
        mask = ids == dataset_id
        if not np.any(mask):
            continue
        result[dataset_id] = compute_classification_metrics(y_true[mask], y_pred[mask])
        result[dataset_id]["num_samples"] = int(np.sum(mask))

    return result


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Iterable[str],
    save_path: str,
) -> None:
    """Plot and save confusion matrices."""
    labels = list(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalized = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(history, save_path: Optional[str] = None) -> None:
    """Plot training and validation curves."""
    if save_path is None:
        save_path = os.path.join(OUTPUT_PATH, "training_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history.get("accuracy", []), label="Train", linewidth=2)
    axes[0].plot(history.history.get("val_accuracy", []), label="Validation", linewidth=2)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history.get("loss", []), label="Train", linewidth=2)
    axes[1].plot(history.history.get("val_loss", []), label="Validation", linewidth=2)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Training history plot saved to: {save_path}")


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot per-class precision, recall, and F1."""
    if save_path is None:
        save_path = os.path.join(OUTPUT_PATH, "per_class_metrics.png")

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    x = np.arange(len(CANONICAL_EMOTIONS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision", color="#2ecc71")
    ax.bar(x, recall, width, label="Recall", color="#3498db")
    ax.bar(x + width, f1, width, label="F1-Score", color="#e74c3c")

    ax.set_xlabel("Emotion")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(CANONICAL_EMOTIONS, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_evaluation(
    model_path: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
    protocol: str = "random",
) -> dict:
    """Run default evaluation workflow."""
    if model_path is None:
        model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith("_best.keras")]
        if not model_files:
            model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".keras")]
        if not model_files:
            raise FileNotFoundError(f"No model found in {MODEL_PATH}")
        model_path = os.path.join(MODEL_PATH, sorted(model_files)[-1])

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    if datasets is None:
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
        metrics = evaluate_model(model, X_test, y_test)
        return metrics

    prepared = prepare_multidataset_data(datasets=datasets, protocol=protocol)
    first_key = sorted(prepared.keys())[0]
    split = prepared[first_key]

    metrics = evaluate_model(
        model,
        split["X_test"],
        split["y_test"],
        output_prefix=first_key,
    )
    metrics["per_dataset"] = evaluate_by_dataset(
        metrics["y_true"],
        metrics["y_pred"],
        split["test_dataset_ids"],
    )
    return metrics


if __name__ == "__main__":
    run_evaluation()
