"""Modern tf.keras LSTM forecaster with leakage-safe windowing and evaluation."""
from __future__ import annotations

import numpy as np


def make_windows(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if lookback < 1 or len(values) <= lookback:
        raise ValueError("lookback must be positive and smaller than the series length")
    x = np.stack([values[i : i + lookback] for i in range(len(values) - lookback)])
    y = values[lookback:]
    return x[..., None], y[..., None]


def build_model(lookback: int, units: int = 64, dropout: float = 0.1):
    """Build a compiled model; TensorFlow is imported lazily for lightweight tooling."""
    if lookback < 1 or units < 1 or not 0 <= dropout < 1:
        raise ValueError("invalid model hyperparameters")
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, 1)),
            tf.keras.layers.LSTM(units, return_sequences=False),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape or not len(y_true):
        raise ValueError("predictions and targets must have the same non-empty shape")
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(error) / denom) * 100)
    return {"mae": mae, "rmse": rmse, "mape_percent": mape}
