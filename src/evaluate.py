"""Evaluation and visualization module."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras.models import Model, load_model

from src.config import OUTPUT_PATH, EMOTIONS, MODEL_PATH
from src.data_loader import prepare_data, get_emotion_name


def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_plots: bool = True
) -> dict:
    """
    Evaluate model and generate metrics.

    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        save_plots: Whether to save plots to disk

    Returns:
        Dictionary containing evaluation metrics
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Print results
    print("\n" + "=" * 60)
    print("Model Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Classification report
    emotion_names = [EMOTIONS[f'{i+1:02d}'] for i in range(len(EMOTIONS))]
    report = classification_report(y_true, y_pred, target_names=emotion_names)
    print("\nClassification Report:")
    print(report)

    # Generate plots
    if save_plots:
        plot_confusion_matrix(y_true, y_pred, emotion_names)
        print(f"\nPlots saved to: {OUTPUT_PATH}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Normalized
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix.png'), dpi=300)
    plt.close()


def plot_training_history(history, save_path: str = None) -> None:
    """
    Plot training and validation curves.

    Args:
        history: Keras training history
        save_path: Path to save the plot
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_PATH, 'training_history.png')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Training history plot saved to: {save_path}")


def plot_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot per-class precision, recall, and F1-score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    emotion_names = [EMOTIONS[f'{i+1:02d}'] for i in range(len(EMOTIONS))]

    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    x = np.arange(len(emotion_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    ax.set_xlabel('Emotion')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'per_class_metrics.png'), dpi=300)
    plt.close()


def run_evaluation(model_path: str = None) -> dict:
    """
    Run complete evaluation pipeline.

    Args:
        model_path: Path to saved model (uses best model if None)

    Returns:
        Evaluation metrics dictionary
    """
    # Find model
    if model_path is None:
        # Look for best model in models directory
        model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('_best.keras')]
        if not model_files:
            model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.keras')]
        if not model_files:
            raise FileNotFoundError(f"No model found in {MODEL_PATH}")
        model_path = os.path.join(MODEL_PATH, sorted(model_files)[-1])

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # Load test data
    print("Loading test data...")
    _, _, X_test, _, _, y_test = prepare_data()

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Additional plots
    plot_per_class_metrics(metrics['y_true'], metrics['y_pred'])

    return metrics


if __name__ == "__main__":
    run_evaluation()
