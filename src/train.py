"""Training pipeline for speech emotion recognition."""

import os
from datetime import datetime
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.models import Model

from src.config import MODEL_PATH, OUTPUT_PATH, BATCH_SIZE, EPOCHS
from src.data_loader import prepare_data
from src.model import create_cnn_lstm_model, get_model_summary


def get_callbacks(model_name: str) -> list:
    """
    Create training callbacks.

    Args:
        model_name: Base name for saved model

    Returns:
        List of Keras callbacks
    """
    # Ensure directories exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),

        # Model checkpoint
        ModelCheckpoint(
            filepath=os.path.join(MODEL_PATH, f'{model_name}_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),

        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(OUTPUT_PATH, 'logs', model_name),
            histogram_freq=1
        )
    ]

    return callbacks


def train_model(
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    model_name: str = None
) -> tuple:
    """
    Complete training pipeline.

    Args:
        batch_size: Training batch size
        epochs: Maximum number of epochs
        model_name: Name for saved model (auto-generated if None)

    Returns:
        Tuple of (trained model, training history)
    """
    # Generate model name with timestamp
    if model_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'emotion_cnn_lstm_{timestamp}'

    print("=" * 60)
    print("Speech Emotion Recognition - Training")
    print("=" * 60)

    # Load and prepare data
    print("\n1. Loading and preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")

    # Create model
    print("\n2. Creating CNN-LSTM model...")
    model = create_cnn_lstm_model()
    print(get_model_summary(model))

    # Get callbacks
    callbacks = get_callbacks(model_name)

    # Train model
    print("\n3. Training model...")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    print()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(MODEL_PATH, f'{model_name}_final.keras')
    model.save(final_model_path)
    print(f"\n4. Model saved to: {final_model_path}")

    # Print final metrics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

    return model, history, (X_test, y_test)


if __name__ == "__main__":
    train_model()
