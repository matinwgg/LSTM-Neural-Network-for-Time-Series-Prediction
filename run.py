"""Train and evaluate the LSTM time-series experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

from core.data_processor import DataLoader
from core.model import Model


def plot_results(predicted_data, true_data, output_path=None, show=False):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(facecolor="white")
    ax.plot(true_data, label="True data")
    ax.plot(predicted_data, label="Prediction")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalised value")
    ax.legend()
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_results_multiple(
    predicted_data, true_data, prediction_len, output_path=None, show=False
):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(facecolor="white")
    ax.plot(true_data, label="True data")
    for index, data in enumerate(predicted_data):
        padding = [None] * (index * prediction_len)
        ax.plot(padding + data, label=f"Prediction {index + 1}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalised value")
    ax.legend()
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"[Plot] Saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def load_config(path) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="JSON config file")
    parser.add_argument("--data-dir", default="data", help="directory containing the CSV")
    parser.add_argument("--filename", help="override configs['data']['filename']")
    parser.add_argument(
        "--columns",
        nargs="+",
        help="override the feature columns (the first column is the target)",
    )
    parser.add_argument("--sequence-length", type=int, help="override window length")
    parser.add_argument("--split", type=float, help="override chronological train split")
    parser.add_argument("--epochs", type=int, help="override training epochs")
    parser.add_argument("--batch-size", type=int, help="override batch size")
    parser.add_argument("--save-dir", help="override the model output directory")
    parser.add_argument(
        "--model",
        help="load an existing model instead of training (prediction-only mode)",
    )
    parser.add_argument(
        "--normalise",
        dest="normalise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable per-window normalisation",
    )
    parser.add_argument(
        "--plot-output",
        help="PNG output path (default: <save-dir>/predictions.png)",
    )
    parser.add_argument(
        "--predictions-output",
        help="optional CSV path for point-by-point predictions",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the plot window instead of only writing a PNG",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip plotting entirely"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configs = load_config(args.config)
    data_config = configs.setdefault("data", {})
    model_config = configs.setdefault("model", {})
    training_config = configs.setdefault("training", {})

    filename = args.filename or data_config.get("filename", "sp500.csv")
    columns = args.columns or data_config.get("columns", ["Close"])
    sequence_length = args.sequence_length or int(data_config.get("sequence_length", 50))
    split = args.split if args.split is not None else float(
        data_config.get("train_test_split", 0.85)
    )
    normalise = (
        args.normalise
        if args.normalise is not None
        else bool(data_config.get("normalise", True))
    )
    epochs = args.epochs or int(training_config.get("epochs", 2))
    batch_size = args.batch_size or int(training_config.get("batch_size", 32))
    save_dir = args.save_dir or model_config.get("save_dir", "saved_models")

    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(args.seed)
    except ImportError:  # TensorFlow is required only when training/predicting.
        pass

    # Keep the configured input contract in sync with CLI overrides.
    for layer in model_config.get("layers", []):
        if str(layer.get("type", "")).lower() == "lstm":
            layer["input_timesteps"] = sequence_length - 1
            layer["input_dim"] = len(columns)
            break

    data = DataLoader(
        os.path.join(args.data_dir, filename),
        split,
        columns,
    )

    if args.model:
        model = Model()
        model.load_model(args.model)
    else:
        model = Model()
        model.build_model(configs)
        steps_per_epoch = max(
            1,
            math.ceil(
                (data.len_train - sequence_length) / batch_size
            ),
        )
        model.train_generator(
            data_gen=data.generate_train_batch(
                seq_len=sequence_length,
                batch_size=batch_size,
                normalise=normalise,
            ),
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            save_dir=save_dir,
        )

    x_test, y_test = data.get_test_data(
        seq_len=sequence_length,
        normalise=normalise,
    )
    true_data = y_test.reshape(-1)

    point_predictions = model.predict_point_by_point(x_test)
    rmse = float(np.sqrt(np.mean((point_predictions - true_data) ** 2)))
    mae = float(np.mean(np.abs(point_predictions - true_data)))
    print(f"[Evaluation] RMSE={rmse:.6f} MAE={mae:.6f}")

    if args.predictions_output:
        output = Path(args.predictions_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output,
            np.column_stack([true_data, point_predictions]),
            delimiter=",",
            header="actual,prediction",
            comments="",
        )
        print(f"[Evaluation] Predictions written to {output}")

    if not args.no_plot:
        predictions = model.predict_sequences_multiple(
            x_test, sequence_length, sequence_length
        )
        plot_output = args.plot_output or str(
            Path(save_dir) / "predictions.png"
        )
        plot_results_multiple(
            predictions,
            true_data,
            sequence_length,
            output_path=plot_output,
            show=args.show,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
