import numpy as np
import pytest

from src.lstm_forecaster import make_windows, regression_metrics


def test_make_windows_shapes_and_alignment():
    x, y = make_windows(np.arange(6), 3)
    assert x.shape == (3, 3, 1)
    assert y.shape == (3, 1)
    np.testing.assert_array_equal(x[:, :, 0], [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(y[:, 0], [3, 4, 5])


def test_make_windows_rejects_invalid_lookback():
    with pytest.raises(ValueError):
        make_windows(np.arange(3), 3)


def test_regression_metrics():
    metrics = regression_metrics(np.array([1, 2, 4]), np.array([1, 3, 2]))
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["mape_percent"] == pytest.approx((0 + 0.5 + 0.5) / 3 * 100)
