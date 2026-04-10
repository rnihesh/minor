"""Model builders for baseline and robustness-focused SER variants."""

from __future__ import annotations

from typing import Callable

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
    SeparableConv1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from src.config import LEARNING_RATE, MAX_LEN, N_MFCC, NUM_CLASSES


def categorical_focal_loss(alpha: float = 0.25, gamma: float = 2.0) -> Callable:
    """Create categorical focal loss function."""

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        modulating = alpha * tf.pow(1.0 - y_pred, gamma)
        loss = tf.reduce_sum(modulating * ce, axis=-1)
        return tf.reduce_mean(loss)

    return loss_fn


def _compile_model(
    model: Model,
    learning_rate: float,
    use_focal_loss: bool = False,
) -> Model:
    loss = categorical_focal_loss() if use_focal_loss else "categorical_crossentropy"
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def create_cnn_lstm_baseline(
    input_shape: tuple,
    num_classes: int,
    learning_rate: float,
    use_focal_loss: bool = False,
) -> Model:
    """Baseline CNN-LSTM architecture."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(64, kernel_size=5, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=5, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(256, kernel_size=3, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(96, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return _compile_model(model, learning_rate=learning_rate, use_focal_loss=use_focal_loss)


def create_attention_cnn_lstm(
    input_shape: tuple,
    num_classes: int,
    learning_rate: float,
    use_focal_loss: bool = False,
) -> Model:
    """Attention-enhanced CNN-LSTM architecture."""
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)

    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="attention_cnn_lstm")
    return _compile_model(model, learning_rate=learning_rate, use_focal_loss=use_focal_loss)


def create_lightweight_cnn_lstm(
    input_shape: tuple,
    num_classes: int,
    learning_rate: float,
    use_focal_loss: bool = False,
) -> Model:
    """Lightweight regularized variant for lower compute budgets."""
    inputs = Input(shape=input_shape)

    x = SeparableConv1D(48, kernel_size=5, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = SeparableConv1D(96, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = LSTM(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(x)
    x = LSTM(48, dropout=0.2, recurrent_dropout=0.1)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="lightweight_cnn_lstm")
    return _compile_model(model, learning_rate=learning_rate, use_focal_loss=use_focal_loss)


def create_model(
    variant: str = "baseline",
    input_shape: tuple = (MAX_LEN, N_MFCC),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    use_focal_loss: bool = False,
) -> Model:
    """Factory for all supported model variants."""
    key = variant.strip().lower()
    if key in {"baseline", "cnn_lstm", "cnn-lstm"}:
        return create_cnn_lstm_baseline(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
            use_focal_loss=use_focal_loss,
        )
    if key in {"attention", "attention_cnn_lstm", "attention-cnn-lstm"}:
        return create_attention_cnn_lstm(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
            use_focal_loss=use_focal_loss,
        )
    if key in {"lightweight", "light", "lite"}:
        return create_lightweight_cnn_lstm(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=learning_rate,
            use_focal_loss=use_focal_loss,
        )
    raise ValueError("Unsupported model variant. Use baseline, attention, or lightweight.")


def create_cnn_lstm_model(
    input_shape: tuple = (MAX_LEN, N_MFCC),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """Backward-compatible alias for the original baseline model."""
    return create_model(
        variant="baseline",
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
        use_focal_loss=False,
    )


def get_model_summary(model: Model) -> str:
    """Get model summary as string."""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)
