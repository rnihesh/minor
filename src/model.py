"""CNN-LSTM model architecture for speech emotion recognition."""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization,
    LSTM, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam

from src.config import N_MFCC, MAX_LEN, NUM_CLASSES, LEARNING_RATE


def create_cnn_lstm_model(
    input_shape: tuple = (MAX_LEN, N_MFCC),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE
) -> Model:
    """
    Create CNN-LSTM hybrid model for emotion recognition.

    Architecture:
        CNN Block (Spatial Features):
        - Conv1D(64) + BatchNorm + ReLU + MaxPool
        - Conv1D(128) + BatchNorm + ReLU + MaxPool
        - Conv1D(256) + BatchNorm + ReLU + MaxPool

        LSTM Block (Temporal Features):
        - LSTM(128, return_sequences=True) + Dropout
        - LSTM(64) + Dropout

        Classification:
        - Dense(64, relu) + Dropout
        - Dense(num_classes, softmax)

    Args:
        input_shape: Input tensor shape (time_steps, features)
        num_classes: Number of emotion classes
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Input
        Input(shape=input_shape),

        # CNN Block 1
        Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # CNN Block 2
        Conv1D(128, kernel_size=5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # CNN Block 3
        Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # LSTM Block 1
        LSTM(128, return_sequences=True),
        Dropout(0.3),

        # LSTM Block 2
        LSTM(64),
        Dropout(0.3),

        # Classification
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_model_summary(model: Model) -> str:
    """
    Get model summary as string.

    Args:
        model: Keras model

    Returns:
        Model summary string
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)
