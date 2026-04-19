"""Enum-keyed model strategy registry.

Maps :class:`ModelStrategyType` enum members to :class:`ModelStrategyInfo`
dataclasses that bundle the display name, description, strategy class, and
default hyperparameters.  Used by the UI and by :func:`init_forecast` to
instantiate strategies dynamically.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Type, Dict, Any

from .strategies.interfaces import ModelStrategy
from .strategies import (
    LightGBMModelStrategy,
    XGBoostModelStrategy,
    LSTMModelStrategy,
    TransformerModelStrategy,
    HussainTransformerModelStrategy,
    HybridTransformerLSTMModelStrategy,
)

@dataclass
class ModelStrategyInfo:
    name: str  # Internal ID/Enum value
    display_name: str  # User-friendly name for UI
    description: str  # Tooltip/description
    strategy_class: Type[ModelStrategy]  # The actual class to instantiate
    default_params: Dict[str, Any]  # Default hyperparameters if needed

class ModelStrategyType(Enum):
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    HUSSAIN_TRANSFORMER = "hussain_transformer"
    HYBRID_TRANSFORMER_LSTM = "hybrid_transformer_lstm"

# Registry dictionary
MODEL_STRATEGY_REGISTRY: Dict[ModelStrategyType, ModelStrategyInfo] = {
    ModelStrategyType.LIGHTGBM: ModelStrategyInfo(
        name=ModelStrategyType.LIGHTGBM.value,
        display_name="LightGBM",
        description="Fast, distributed, high-performance gradient boosting framework based on decision tree algorithms.",
        strategy_class=LightGBMModelStrategy,
        default_params={"n_estimators": 100, "learning_rate": 0.1}
    ),
    ModelStrategyType.XGBOOST: ModelStrategyInfo(
        name=ModelStrategyType.XGBOOST.value,
        display_name="XGBoost",
        description="Optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.",
        strategy_class=XGBoostModelStrategy,
        default_params={"n_estimators": 100, "max_depth": 3}
    ),
    ModelStrategyType.LSTM: ModelStrategyInfo(
        name=ModelStrategyType.LSTM.value,
        display_name="LSTM",
        description="Long Short-Term Memory recurrent neural network for sequence-based time-series forecasting.",
        strategy_class=LSTMModelStrategy,
        default_params={"epochs": 50, "batch_size": 32}
    ),
    ModelStrategyType.TRANSFORMER: ModelStrategyInfo(
        name=ModelStrategyType.TRANSFORMER.value,
        display_name="Transformer",
        description="Attention-based Transformer architecture for time-series forecasting.",
        strategy_class=TransformerModelStrategy,
        default_params={"epochs": 50, "batch_size": 32}
    ),
    ModelStrategyType.HYBRID_TRANSFORMER_LSTM: ModelStrategyInfo(
        name=ModelStrategyType.HYBRID_TRANSFORMER_LSTM.value,
        display_name="Hybrid Transformer-LSTM",
        description="Hybrid architecture combining Transformer attention with LSTM sequential memory.",
        strategy_class=HybridTransformerLSTMModelStrategy,
        default_params={"epochs": 50, "batch_size": 32}
    ),
    ModelStrategyType.HUSSAIN_TRANSFORMER: ModelStrategyInfo(
        name=ModelStrategyType.HUSSAIN_TRANSFORMER.value,
        display_name="Hussain Transformer",
        description="Simplified single-attention Transformer from Hussain et al. (2025) for EV charging demand forecasting.",
        strategy_class=HussainTransformerModelStrategy,
        default_params={"epochs": 100, "batch_size": 32, "dropout": 0.2}
    ),
}

def get_model_strategy(strategy_type: ModelStrategyType) -> ModelStrategy:
    """Factory function to get a model strategy instance."""
    strategy_info = MODEL_STRATEGY_REGISTRY.get(strategy_type)
    if not strategy_info:
        raise ValueError(f"Unknown model strategy: {strategy_type}")
    return strategy_info.strategy_class()
