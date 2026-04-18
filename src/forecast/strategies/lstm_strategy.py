"""
LSTM recurrent neural network strategy for time-series forecasting.

Architecture (Sequential):
  Input → LSTM(64, return_sequences=True) → Dropout →
          LSTM(32) → Dropout → Dense(1)

Hyperparameters read from ``df.attrs['lstm_params']``:
  epochs, batch_size, learning_rate, activation, dropout_rate, look_back,
  train_range, test_range, forecast_horizon, etc.

All shared train/predict pipeline logic lives in ``KerasTimeSeriesBaseStrategy``.
"""

from typing import Tuple

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Input, LSTM as KerasLSTM, Dense, Dropout

from .base_keras import KerasTimeSeriesBaseStrategy


class LSTMModelStrategy(KerasTimeSeriesBaseStrategy):
    """Two-layer LSTM with dropout for univariate time-series forecasting."""

    @property
    def _params_key(self) -> str:
        return 'lstm_params'

    @property
    def _strategy_display_name(self) -> str:
        return 'LSTM'

    def build_model(
        self,
        input_shape: Tuple[int, ...],
        learning_rate: float = 0.001,
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        output_steps: int = 1,
        **_kwargs,
    ):
        """Build a two-layer LSTM model.

        Parameters
        ----------
        input_shape : tuple
            ``(look_back, 1)`` for univariate data.
        learning_rate : float
            Adam optimiser learning rate.
        activation : str
            Activation function for LSTM cells.
        dropout_rate : float
            Fraction of units to drop after each LSTM layer.
        output_steps : int
            Number of output values.  1 for single-step (backtest) models;
            ``forecast_horizon`` for direct multi-step models.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(KerasLSTM(units=64, activation=activation, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(KerasLSTM(units=32, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_steps))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model
