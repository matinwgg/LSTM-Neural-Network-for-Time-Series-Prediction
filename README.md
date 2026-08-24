# LSTM Neural Network for Time-Series Prediction

An implementation-oriented study of recurrent neural networks and Long Short-Term Memory (LSTM) models for sequential and time-series prediction.

## What it demonstrates

- sequence/window construction
- recurrent state and temporal dependencies
- LSTM gating intuition
- multi-step and sequence prediction
- sine-wave experiments
- multidimensional stock-market time-series experiments
- practical preprocessing and normalization

## Mathematics behind the model

The project connects recurrent learning to matrix multiplication, vector states, nonlinear activation functions, recurrence relations, gradient-based optimization, the chain rule, backpropagation through time, numerical stability, and sequence statistics.

A useful abstraction is:

`h_t = f(x_t, h_{t-1}; θ)`

where the hidden state is a recursively updated representation of the sequence history. LSTM gates regulate how information is retained, forgotten, and exposed over time.

## Numerical considerations

Time-series normalization must handle zero-valued baselines and non-finite values. Training and evaluation should avoid leakage from future observations into preprocessing statistics. Sequence length, batch size, learning rate, and recurrent-state initialization all affect optimization and generalization.

## Historical implementation note

The original project uses an older TensorFlow/Keras stack. Treat the supplied environment as a reproducibility target rather than a recommendation for a new production system.

## References

- [Original article](https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks)
- [Video walkthrough](https://www.youtube.com/watch?v=2np77NOdnwk)

## Status

**Machine-learning study project.** The repository is intended for learning, experimentation, and understanding recurrent neural-network mathematics.
