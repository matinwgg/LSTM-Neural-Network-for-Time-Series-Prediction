"""Keras LSTM model used by the time-series experiments."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np

try:  # Import lazily so the data utilities work without TensorFlow installed.
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except Exception as exc:  # pragma: no cover - depends on the local environment
    keras = None
    layers = None
    EarlyStopping = None
    ModelCheckpoint = None
    _TF_IMPORT_ERROR = exc
else:
    _TF_IMPORT_ERROR = None


def require_tensorflow() -> None:
    if keras is None:
        raise RuntimeError(
            "TensorFlow is required for training and prediction. "
            "Install a supported version with `pip install -r requirements.txt`."
        ) from _TF_IMPORT_ERROR


class Model:
    """Build, train, save, load, and run an LSTM forecaster."""

    def __init__(self):
        self.model = None

    def build_model(self, configs: dict):
        require_tensorflow()
        model_config = configs.get("model", {})
        layer_configs = model_config.get("layers")
        if not layer_configs:
            raise ValueError("configs['model']['layers'] must contain at least one layer")

        model = keras.Sequential(name="lstm_forecaster")
        for index, layer in enumerate(layer_configs):
            layer_type = str(layer.get("type", "")).lower()
            if layer_type == "lstm":
                neurons = int(layer.get("neurons", 0))
                if neurons <= 0:
                    raise ValueError("LSTM layers require a positive 'neurons' value")
                kwargs = {
                    "return_sequences": bool(layer.get("return_seq", False)),
                    "activation": layer.get("activation", "tanh"),
                }
                input_timesteps = layer.get("input_timesteps")
                input_dim = layer.get("input_dim")
                if input_timesteps is not None or input_dim is not None:
                    if not input_timesteps or not input_dim:
                        raise ValueError(
                            "input_timesteps and input_dim must be provided together"
                        )
                    kwargs["input_shape"] = (int(input_timesteps), int(input_dim))
                if index != 0 and "input_shape" in kwargs:
                    # Only the first layer defines the input contract.
                    kwargs.pop("input_shape")
                model.add(layers.LSTM(neurons, **kwargs))
            elif layer_type == "dropout":
                rate = float(layer.get("rate", 0.0))
                if not 0.0 <= rate < 1.0:
                    raise ValueError("dropout rate must be in [0, 1)")
                model.add(layers.Dropout(rate))
            elif layer_type == "dense":
                neurons = int(layer.get("neurons", 0))
                if neurons <= 0:
                    raise ValueError("Dense layers require a positive 'neurons' value")
                model.add(
                    layers.Dense(
                        neurons,
                        activation=layer.get("activation", "linear"),
                    )
                )
            else:
                raise ValueError(f"unsupported layer type: {layer_type!r}")

        loss = model_config.get("loss", "mse")
        optimizer = model_config.get("optimizer", "adam")
        metrics = model_config.get("metrics", ["mae"])
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.model = model
        print("[Model] Model compiled")
        return self.model

    def load_model(self, filepath, compile: bool = False):
        require_tensorflow()
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"model not found: {path}")
        print(f"[Model] Loading model from {path}")
        self.model = keras.models.load_model(path, compile=compile)
        return self.model

    def train(
        self,
        x,
        y,
        epochs: int,
        batch_size: int,
        save_dir,
        validation_split: float = 0.1,
        callbacks=None,
    ):
        require_tensorflow()
        self._require_built()
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("x and y must be non-empty and have equal lengths")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")

        save_fname = self._save_path(save_dir, epochs)
        monitor = "val_loss" if validation_split > 0 else "loss"
        train_callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=2,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=save_fname,
                monitor=monitor,
                save_best_only=True,
            ),
        ]
        if callbacks:
            train_callbacks.extend(callbacks)

        print(f"[Model] Training for {epochs} epoch(s), batch size {batch_size}")
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=train_callbacks,
            verbose=1,
        )
        self.model.save(save_fname)
        print(f"[Model] Training complete; model saved to {save_fname}")
        return save_fname

    def train_generator(
        self,
        data_gen,
        epochs: int,
        batch_size: int,
        steps_per_epoch: int,
        save_dir,
        callbacks=None,
    ):
        require_tensorflow()
        self._require_built()
        if epochs <= 0 or batch_size <= 0 or steps_per_epoch <= 0:
            raise ValueError("epochs, batch_size, and steps_per_epoch must be positive")

        save_fname = self._save_path(save_dir, epochs)
        train_callbacks = [
            EarlyStopping(
                monitor="loss",
                patience=2,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=save_fname,
                monitor="loss",
                save_best_only=True,
            ),
        ]
        if callbacks:
            train_callbacks.extend(callbacks)

        print(
            f"[Model] Training for {epochs} epoch(s), batch size {batch_size}, "
            f"{steps_per_epoch} batch(es) per epoch"
        )
        self.model.fit(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=train_callbacks,
            verbose=1,
            workers=1,
            use_multiprocessing=False,
        )
        self.model.save(save_fname)
        print(f"[Model] Training complete; model saved to {save_fname}")
        return save_fname

    def predict_point_by_point(self, data, batch_size: int = 256):
        require_tensorflow()
        self._require_built()
        data = np.asarray(data, dtype=float)
        if data.size == 0:
            raise ValueError("prediction data must not be empty")
        print("[Model] Predicting point-by-point")
        predicted = self.model.predict(data, batch_size=batch_size, verbose=0)
        return np.asarray(predicted).reshape(-1)

    def predict_sequences_multiple(
        self, data, window_size: int, prediction_len: int
    ):
        require_tensorflow()
        self._require_built()
        data = np.asarray(data, dtype=float)
        if data.ndim != 3 or len(data) == 0:
            raise ValueError("data must be a non-empty (windows, timesteps, features) array")
        if prediction_len <= 0:
            raise ValueError("prediction_len must be positive")

        print("[Model] Predicting sequences multiple")
        prediction_seqs = []
        for i in range(0, len(data), prediction_len):
            prediction_seqs.append(
                self._autoregressive_predict(data[i], prediction_len)
            )
        return prediction_seqs

    def predict_sequence_full(self, data, window_size: int):
        require_tensorflow()
        self._require_built()
        data = np.asarray(data, dtype=float)
        if data.ndim != 3 or len(data) == 0:
            raise ValueError("data must be a non-empty (windows, timesteps, features) array")
        print("[Model] Predicting a full sequence")
        return self._autoregressive_predict(data[0], len(data))

    def evaluate(self, x, y, batch_size: int = 256):
        require_tensorflow()
        self._require_built()
        return self.model.evaluate(x, y, batch_size=batch_size, verbose=0)

    def _autoregressive_predict(self, frame, prediction_len: int):
        frame = np.asarray(frame, dtype=float)
        if frame.ndim != 2 or frame.shape[0] < 2:
            raise ValueError("each prediction window must have at least two timesteps")
        predicted = []
        for _ in range(prediction_len):
            value = float(self.model.predict(frame[np.newaxis, :, :], verbose=0)[0, 0])
            predicted.append(value)
            next_row = frame[-1].copy()
            # The model forecasts the first feature (typically Close). Carry the
            # remaining features forward rather than overwriting them with the
            # predicted value, which would corrupt multivariate inputs.
            next_row[0] = value
            frame = np.vstack([frame[1:], next_row])
        return predicted

    def _require_built(self) -> None:
        if self.model is None:
            raise RuntimeError("call build_model() or load_model() first")

    @staticmethod
    def _save_path(save_dir, epochs: int) -> str:
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%d%m%Y-%H%M%S-%f")
        return str(directory / f"{timestamp}-e{epochs}.keras")
