import numpy as np
import pandas as pd
import pytest

from core.data_processor import DataLoader


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "series.csv"
    rows = 100
    pd.DataFrame(
        {
            "Close": np.arange(1, rows + 1, dtype=float),
            "Volume": np.arange(101, rows + 101, dtype=float),
        }
    ).to_csv(path, index=False)
    return path


def test_split_shapes_and_normalisation(dataset):
    loader = DataLoader(dataset, split=0.8, cols=["Close", "Volume"])
    assert loader.len_train == 80
    assert loader.len_test == 20

    x_train, y_train = loader.get_train_data(seq_len=10, normalise=True)
    assert x_train.shape == (70, 9, 2)
    assert y_train.shape == (70, 1)
    assert np.isfinite(x_train).all()
    assert np.isfinite(y_train).all()

    x_test, y_test = loader.get_test_data(seq_len=10, normalise=True)
    assert x_test.shape == (10, 9, 2)
    assert y_test.shape == (10, 1)
    assert np.isfinite(x_test).all()
    assert np.isfinite(y_test).all()


def test_unnormalised_windows_keep_input_values(dataset):
    loader = DataLoader(dataset, split=0.8, cols=["Close"])
    x, y = loader.get_train_data(seq_len=3, normalise=False)
    np.testing.assert_allclose(x[0, :, 0], [1.0, 2.0])
    np.testing.assert_allclose(y[0], [3.0])


def test_generator_yields_partial_final_batch(dataset):
    loader = DataLoader(dataset, split=0.8, cols=["Close", "Volume"])
    batches = list(loader.generate_train_batch(seq_len=10, batch_size=25, normalise=False))
    assert [len(x) for x, _ in batches] == [25, 25, 20]
    assert all(x.shape[1:] == (9, 2) for x, _ in batches)


def test_missing_columns_are_rejected(dataset):
    with pytest.raises(ValueError, match="missing"):
        DataLoader(dataset, split=0.8, cols=["Close", "DoesNotExist"])


def test_invalid_split_is_rejected(dataset):
    with pytest.raises(ValueError, match="split"):
        DataLoader(dataset, split=1.0, cols=["Close"])


def test_non_numeric_values_are_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Close": [1.0, "oops", 3.0, 4.0, 5.0, 6.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="missing or non-numeric"):
        DataLoader(path, split=0.5, cols=["Close"])


def test_zero_baseline_is_rejected(tmp_path):
    path = tmp_path / "zero.csv"
    pd.DataFrame({"Close": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}).to_csv(
        path, index=False
    )
    loader = DataLoader(path, split=0.5, cols=["Close"])
    with pytest.raises(ValueError, match="zero base"):
        loader.get_train_data(seq_len=2, normalise=True)


def test_short_test_set_is_rejected(dataset):
    loader = DataLoader(dataset, split=0.9, cols=["Close"])
    with pytest.raises(ValueError, match="more than seq_len"):
        loader.get_test_data(seq_len=10, normalise=False)


def test_invalid_sequence_length_is_rejected(dataset):
    loader = DataLoader(dataset, split=0.8, cols=["Close"])
    with pytest.raises(ValueError, match="seq_len"):
        loader.get_train_data(seq_len=1, normalise=False)


def test_single_window_normalisation_shape(dataset):
    loader = DataLoader(dataset, split=0.8, cols=["Close"])
    window = loader.data_train[:5]
    result = loader.normalise_windows(window, single_window=True)
    assert result.shape == (1, 5, 1)
    assert np.isfinite(result).all()
