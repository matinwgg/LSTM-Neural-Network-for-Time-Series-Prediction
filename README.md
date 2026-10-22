# LSTM Neural Network for Time-Series Prediction

## About

An implementation-oriented study of recurrent neural networks and Long
Short-Term Memory (LSTM) models for sequential and time-series prediction. The
repository contains the model, chronological window construction, training and
prediction scripts, and synthetic/financial datasets used by the experiments.

## Features

- Sliding-window construction with explicit train/test separation
- Per-window normalisation with zero/non-finite base validation
- Stacked LSTM, dropout, and dense output layers
- Generator-based training for large datasets
- Point-by-point and autoregressive sequence prediction
- Headless plot output for CI and servers
- Configurable via JSON and command-line overrides
- Tests for preprocessing edge cases and model configuration

## Tech Stack

- Python 3.10-3.12 (TensorFlow may lag the newest Python release)
- TensorFlow 2.x / Keras 3
- NumPy, pandas, Matplotlib

## Architecture

```text
CSV data
   ↓
numeric validation + chronological split
   ↓
sliding windows + per-window normalisation
   ↓
stacked LSTM
   ↓
point-by-point / autoregressive prediction
   ↓
RMSE/MAE + plot
```

## Project Structure

```text
.
├── core/
│   ├── data_processor.py  # loading, splitting, windows, normalisation
│   ├── model.py           # Keras model build/train/predict
│   └── utils.py
├── data/                  # sp500.csv and sinewave.csv
├── tests/
├── config.json            # S&P 500 experiment
├── config.sinewave.json   # univariate sine-wave experiment
├── run.py                 # CLI entry point
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/matinwgg/LSTM-Neural-Network-for-Time-Series-Prediction.git
cd LSTM-Neural-Network-for-Time-Series-Prediction

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is a large dependency. If you only want to run the preprocessing
tests, install the lightweight test dependencies:

```bash
pip install "numpy>=1.26,<3" "pandas>=2.1,<3" pytest
pytest
```

## Usage

Train the S&P 500 experiment:

```bash
python run.py
```

The default run writes the best checkpoint and `saved_models/predictions.png`
without opening a window. Use `--show` to display the plot.

Train the sine-wave experiment:

```bash
python run.py --config config.sinewave.json
```

Common overrides:

```bash
python run.py \
  --epochs 10 \
  --batch-size 64 \
  --sequence-length 60 \
  --columns Close Volume \
  --predictions-output saved_models/predictions.csv
```

Prediction-only mode with an existing checkpoint:

```bash
python run.py --model saved_models/19082026-120000-000000-e10.keras \
  --no-plot
```

See all options with `python run.py --help`.

## Data Format

The CSV must contain the requested columns and numeric values. The first
requested column is the prediction target; additional columns are used as
multivariate inputs. Rows must be in chronological order. Missing,
non-numeric, zero-baseline, and non-finite values are rejected with an
explanatory error instead of being silently converted.

Example:

```csv
Date,Open,High,Low,Close,Volume
03-01-00,1469.25,1478,1438.359985,1455.219971,931800000
```

## Evaluation

The script reports point-by-point RMSE and MAE on the held-out chronological
test period. Normalised values are used by the default configuration, so
metrics are relative to the first value of each test window. For absolute
price errors, extend the loader to retain a fitted scaler and invert predictions
before measuring error.

## Mathematics

The core recurrence is `h_t = f(x_t, h_{t-1}; θ)`. LSTM gates control state
transitions through sigmoid and elementwise operations. Training uses the chain
rule and backpropagation through time; numerical behaviour depends on sequence
length, initialisation, learning rate, and activation saturation.

## Testing

```bash
pytest
flake8 core tests run.py --select=E9,F63,F7,F82
bandit -q -ll -r core run.py
```

CI installs only the lightweight NumPy/pandas test dependencies. Tests that
require TensorFlow are skipped automatically when it is not installed.

## Limitations & Future Work

- Per-window normalisation is convenient for this experiment but is not a
  deployment-grade scaler; fit a train-only scaler for production use.
- The autoregressive multi-step path carries non-target features forward. A
  production multivariate model should forecast or condition on those features.
- Add walk-forward evaluation, prediction intervals, and naive baselines.
- Compare against GRU/Transformer models and run systematic hyperparameter
  searches.
- Pin a reproducible lockfile for experiments and record dataset hashes.

## References

- [Original article](https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks)
- [Video walkthrough](https://www.youtube.com/watch?v=2np77NOdnwk)

## Contributing

Preserve chronological evaluation, document preprocessing, and add tests for
numerical edge cases and data leakage.

## License

See repository license information.

## Author

**A. Matin Odoom**
