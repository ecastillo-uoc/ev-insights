"""
Article-faithful LSTM strategy reproducing Hussain et al. (2025).

    Hussain, A., Eswarakrishnan, V., Aslam, A. & Tripura, S.
    "Charging stations demand forecasting using LSTM based hybrid transformer model."
    Sci Rep 15, 13555 (2025). https://doi.org/10.1038/s41598-025-20421-y

Architecture identical to ``LSTMModelStrategy`` (two-layer LSTM with dropout
and a Dense(1) head), but uses the article-specified scaler:

Article hyperparameters (Table 1):
  epochs=100, batch_size=32, learning_rate=0.001,
  activation=ReLU, dropout=0.2, scaler=MinMaxScaler.
  look_back = forecast_horizon (30, 120, or 240 days).

The only behavioural difference from ``LSTMModelStrategy`` is the scaler:
``MinMaxScaler`` instead of ``RobustScaler``.  This ensures faithful
reproduction when comparing against the paper's reported metrics.

All shared train/predict pipeline logic lives in ``KerasTimeSeriesBaseStrategy``.
"""

from sklearn.preprocessing import MinMaxScaler

from .lstm_strategy import LSTMModelStrategy


class HussainLSTMModelStrategy(LSTMModelStrategy):
    """LSTM with Hussain et al. (2025) normalisation: MinMaxScaler instead of RobustScaler."""

    @property
    def _params_key(self) -> str:
        return 'lstm_params'

    @property
    def _strategy_display_name(self) -> str:
        return 'Hussain LSTM'

    @property
    def _scaler_class(self):
        """Article Table 1: scaler=MinMaxScaler."""
        return MinMaxScaler
