"""
Strategy implementations for the ev-insights forecast module.

Re-exports all concrete data and model strategies so consumers can write::

    from src.forecast.strategies import LSTMModelStrategy, SessionDataStrategy

Base classes and utilities:
  * ``interfaces.py`` — ABC definitions (``PredictionTargetStrategy``, ``ModelStrategy``)
  * ``base_keras.py`` — shared train/predict pipeline for Keras models
  * ``utils_ts.py``   — lag and time-feature helpers
"""

# --- Data strategies ---
from .data_strategies import (
    SessionDataStrategy,
    StationChargesDataStrategy,
    StationEnergyDataStrategy,
)

# --- Model strategies ---
from .lightgbm_strategy import LightGBMModelStrategy
from .xgboost_strategy import XGBoostModelStrategy
from .lstm_strategy import LSTMModelStrategy
from .transformer_strategy import TransformerModelStrategy
from .dl_baseline_lstm_strategy import HussainLSTMModelStrategy
from .dl_baseline_transformer_strategy import HussainTransformerModelStrategy
from .dl_baseline_hybrid_strategy import HussainHybridModelStrategy
from .hybrid_strategy import HybridTransformerLSTMModelStrategy

__all__ = [
    # Data strategies
    'SessionDataStrategy',
    'StationChargesDataStrategy',
    'StationEnergyDataStrategy',
    # Model strategies
    'LightGBMModelStrategy',
    'XGBoostModelStrategy',
    'LSTMModelStrategy',
    'TransformerModelStrategy',
    'HussainLSTMModelStrategy',
    'HussainTransformerModelStrategy',
    'HussainHybridModelStrategy',
    'HybridTransformerLSTMModelStrategy',
]
