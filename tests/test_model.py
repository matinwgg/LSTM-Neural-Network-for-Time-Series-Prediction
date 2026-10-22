import pytest

from core.model import Model, keras


CONFIG = {
    "model": {
        "loss": "mse",
        "optimizer": "adam",
        "layers": [
            {
                "type": "lstm",
                "neurons": 4,
                "input_timesteps": 4,
                "input_dim": 1,
                "return_seq": False,
            },
            {"type": "dense", "neurons": 1, "activation": "linear"},
        ],
    }
}


def test_model_fails_clearly_without_tensorflow():
    if keras is not None:
        pytest.skip("TensorFlow is installed; the missing-dependency path is not active")

    with pytest.raises(RuntimeError, match="TensorFlow"):
        Model().build_model(CONFIG)


def test_model_builds_with_tensorflow():
    if keras is None:
        pytest.skip("TensorFlow is not installed")

    model = Model()
    built = model.build_model(CONFIG)
    assert built.output_shape == (None, 1)


def test_unsupported_layer_is_rejected():
    if keras is None:
        pytest.skip("TensorFlow is not installed")

    broken = {
        "model": {
            "layers": [
                {
                    "type": "lstm",
                    "neurons": 4,
                    "input_timesteps": 4,
                    "input_dim": 1,
                },
                {"type": "mystery"},
            ]
        }
    }
    with pytest.raises(ValueError, match="unsupported layer"):
        Model().build_model(broken)


def test_missing_model_file_is_rejected(tmp_path):
    if keras is None:
        pytest.skip("TensorFlow is not installed")

    with pytest.raises(FileNotFoundError):
        Model().load_model(tmp_path / "missing.keras")
