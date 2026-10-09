"""
Deep Convolutional Neural Network for Eye State Classification (OPEN vs. CLOSED).
Custom architecture with BatchNormalization, Dropout, and residual-style conv blocks.
"""

from typing import Tuple, Dict, Any, Optional
import os
import json


def build_eye_cnn(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3
):
    """
    Constructs a robust, production-grade CNN for 64x64 eye patch classification.
    
    Architecture:
    Input (64, 64, 3)
      -> Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout
      -> Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout
      -> Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout
      -> GlobalAveragePooling2D
      -> Dense(128) -> BatchNorm -> ReLU -> Dropout(0.4)
      -> Dense(1, activation='sigmoid')  [Output: 0 = CLOSED, 1 = OPEN]
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers, optimizers

    kernel_reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inputs = layers.Input(shape=input_shape, name="eye_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=kernel_reg, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=kernel_reg, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=kernel_reg, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=kernel_reg, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(dropout_rate * 0.75, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=kernel_reg, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(dropout_rate, name="drop3")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, kernel_regularizer=kernel_reg, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Activation("relu", name="relu_fc1")(x)
    x = layers.Dropout(0.4, name="drop_fc1")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="eye_state_prob")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="driverguard_eye_cnn")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


def get_training_callbacks(
    checkpoint_path: str = "models/eye_classifier.keras",
    patience_es: int = 8,
    patience_lr: int = 3
):
    """
    Returns standard production callbacks for training stabilization:
    EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
    """
    import tensorflow as tf

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


def save_training_metadata(
    output_path: str = "models/model_metadata.json",
    history_dict: Optional[Dict[str, Any]] = None,
    evaluation_dict: Optional[Dict[str, Any]] = None,
    input_shape: Tuple[int, int, int] = (64, 64, 3)
):
    """Saves model training and evaluation metadata to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata = {
        "model_name": "driverguard_eye_cnn",
        "input_shape": list(input_shape),
        "classes": {"0": "CLOSED", "1": "OPEN"},
        "history": history_dict or {},
        "evaluation": evaluation_dict or {},
        "target_metric": "binary_cross_entropy"
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
