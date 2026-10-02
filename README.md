# LSTM Neural Network for Time-Series Prediction

## About

An implementation-oriented study of recurrent neural networks and Long Short-Term Memory (LSTM) models for sequential and time-series prediction. This repository helps in understanding both the model mechanics and the mathematics of learning from temporal data.

### Why it exists

Sequential prediction requires models to represent dependencies across time while managing unstable gradients and changing information relevance. This project uses LSTMs to make those ideas concrete through synthetic and financial time-series experiments.

## Features

- Sequence/window construction
- Recurrent hidden-state modelling
- LSTM gating mechanisms
- Sequence and multi-step prediction experiments
- Sine-wave experiments
- Multidimensional stock-market time-series experiments
- Practical preprocessing and normalisation

## Tech Stack

- Python
- TensorFlow/Keras legacy stack used by the original implementation
- NumPy
- Matplotlib/data tooling used by the experiments

## Architecture

```text
Raw time series
      ↓
Cleaning + normalization
      ↓
Sliding-window sequences
      ↓
LSTM recurrent model
      ↓
Gradient-based training / BPTT
      ↓
Prediction
      ↓
Time-series evaluation
```

## Project Structure

```text
.
├── *.py / notebooks   # Model and experiment code
├── data/              # Input data where applicable
├── models/            # Saved model artefacts where applicable
└── README.md
```

## Prerequisites

Use the historical environment specified by the project's dependency configuration. Modern TensorFlow versions may require code changes because the original implementation uses an older stack.

## Getting Started

```bash
git clone https://github.com/matinwgg/LSTM-Neural-Network-for-Time-Series-Prediction.git
cd LSTM-Neural-Network-for-Time-Series-Prediction
```

Install the project's pinned/compatible dependencies before running the experiment scripts.

## Usage

Typical workflow:

1. Prepare a chronological dataset.
2. Fit preprocessing statistics using training data only.
3. Construct fixed-length windows.
4. Train the LSTM.
5. Evaluate on temporally separated data.
6. Plot predictions and report error metrics.

## Mathematics

The core recurrence is `h_t = f(x_t, h_{t-1}; θ)`. LSTM gates control state transitions through sigmoid and elementwise operations. Training uses the chain rule and backpropagation through time; numerical behaviour depends on sequence length, initialisation, learning rate, and activation saturation.

## Testing / Evaluation

Evaluation should avoid future-data leakage and should report metrics on a held-out chronological period. Preprocessing must handle zero-valued baselines and non-finite values.

## Limitations & Future Work

- Modernise the TensorFlow stack
- Add reproducible environment locking
- Compare against GRU/Transformer baselines
- Add walk-forward evaluation
- Quantify uncertainty
- Add systematic hyperparameter experiments

## References

- [Original article](https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks)
- [Video walkthrough](https://www.youtube.com/watch?v=2np77NOdnwk)

## Contributing

Preserve chronological evaluation, document preprocessing, and add tests for numerical edge cases and data leakage.

## License

See repository license information.

## Author

**A. Matin Odoom**
