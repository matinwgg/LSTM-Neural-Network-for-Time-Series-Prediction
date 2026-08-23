import math
import numpy as np
import pandas as pd

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):
        if not 0 < split < 1:
            raise ValueError("split must be between 0 and 1")
        dataframe = pd.read_csv(filename)
        if not cols or any(col not in dataframe.columns for col in cols):
            raise ValueError("requested columns are missing from the dataset")
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        if seq_len < 2 or self.len_test <= seq_len:
            raise ValueError("test dataset must contain more than seq_len rows")
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        if seq_len < 2 or self.len_train <= seq_len:
            raise ValueError("training dataset must contain more than seq_len rows")
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        if seq_len < 2 or batch_size <= 0:
            raise ValueError("seq_len must be >= 2 and batch_size must be positive")
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    yield np.array(x_batch), np.array(y_batch)
                    return
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        window = self.data_train[i:i+seq_len]
        if len(window) < seq_len:
            raise ValueError("insufficient rows for requested sequence length")
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise each feature relative to its first value in the window.'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                base = float(window[0, col_i])
                if not math.isfinite(base) or base == 0:
                    raise ValueError("cannot normalise a feature with a non-finite or zero base value")
                normalised_col = [((float(p) / base) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T
            if not np.all(np.isfinite(normalised_window)):
                raise ValueError("normalisation produced non-finite values")
            normalised_data.append(normalised_window)
        return np.array(normalised_data)