"""Data loading and sliding-window construction for the LSTM experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class DataLoader:
    """Load a CSV and turn it into chronological train/test windows."""

    def __init__(self, filename, split: float, cols: list[str]):
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"dataset not found: {path}")
        if not 0 < split < 1:
            raise ValueError("split must be between 0 and 1")
        if not cols:
            raise ValueError("at least one column must be requested")

        try:
            dataframe = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - pandas message varies
            raise ValueError(f"could not read dataset: {path}") from exc

        missing = [column for column in cols if column not in dataframe.columns]
        if missing:
            raise ValueError(f"requested columns are missing: {', '.join(missing)}")

        selected = dataframe.loc[:, cols].apply(pd.to_numeric, errors="coerce")
        if selected.isna().any().any():
            raise ValueError("dataset contains missing or non-numeric values")

        values = selected.to_numpy(dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("dataset contains no finite rows")

        i_split = int(len(values) * split)
        if i_split == 0 or i_split == len(values):
            raise ValueError("split must leave both training and test rows")

        self.data_train = values[:i_split]
        self.data_test = values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

        if self.len_train < 2 or self.len_test < 2:
            raise ValueError("training and test sets must each contain at least 2 rows")

    def get_test_data(self, seq_len: int, normalise: bool):
        seq_len = self._validate_seq_len(seq_len)
        if self.len_test <= seq_len:
            raise ValueError("test dataset must contain more than seq_len rows")

        data_windows = np.asarray(
            [self.data_test[i : i + seq_len] for i in range(self.len_test - seq_len)],
            dtype=float,
        )
        if normalise:
            data_windows = self.normalise_windows(data_windows, single_window=False)

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    def get_train_data(self, seq_len: int, normalise: bool):
        seq_len = self._validate_seq_len(seq_len)
        if self.len_train <= seq_len:
            raise ValueError("training dataset must contain more than seq_len rows")

        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.asarray(data_x), np.asarray(data_y)

    def generate_train_batch(self, seq_len: int, batch_size: int, normalise: bool):
        seq_len = self._validate_seq_len(seq_len)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        window_count = self.len_train - seq_len
        if window_count <= 0:
            raise ValueError("training dataset is too short for the requested seq_len")

        i = 0
        while i < window_count:
            x_batch = []
            y_batch = []
            for _ in range(batch_size):
                if i >= window_count:
                    break
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.asarray(x_batch), np.asarray(y_batch)

    def _next_window(self, i: int, seq_len: int, normalise: bool):
        window = self.data_train[i : i + seq_len]
        if len(window) < seq_len:
            raise ValueError("insufficient rows for requested sequence length")
        if normalise:
            window = self.normalise_windows(window, single_window=True)[0]
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    @staticmethod
    def _validate_seq_len(seq_len: int) -> int:
        if not isinstance(seq_len, (int, np.integer)) or seq_len < 2:
            raise ValueError("seq_len must be an integer >= 2")
        return int(seq_len)

    def normalise_windows(self, window_data, single_window: bool = False):
        """Normalise each window relative to its first row.

        Each feature is divided by its own first value in the window and shifted
        down by one. Zero-valued or non-finite bases are rejected rather than
        silently producing infinities.
        """

        windows = np.asarray(window_data, dtype=float)
        if single_window:
            if windows.ndim != 2:
                raise ValueError("single_window expects a 2-D array")
            windows = windows[np.newaxis, :, :]
        if windows.ndim != 3:
            raise ValueError("window_data must be a 2-D or 3-D array")
        if windows.shape[0] == 0 or windows.shape[1] < 2:
            raise ValueError("windows must contain at least two timesteps")

        bases = windows[:, 0, :]
        if not np.isfinite(bases).all():
            raise ValueError("cannot normalise a feature with a non-finite base value")
        if np.any(np.abs(bases) < 1e-12):
            raise ValueError("cannot normalise a feature with a zero base value")

        normalised = windows / bases[:, np.newaxis, :] - 1.0
        if not np.isfinite(normalised).all():
            raise ValueError("normalisation produced non-finite values")
        return normalised
