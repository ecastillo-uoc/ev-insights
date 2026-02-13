from enum import Enum
from dataclasses import dataclass
from typing import Type, Dict, Any

from src.forecast.strategies.interfaces import ModelStrategy
from src.forecast.forecast_implementations import (
    LightGBMModelStrategy,
    XGBoostModelStrategy
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
}

def get_model_strategy(strategy_type: ModelStrategyType) -> ModelStrategy:
    """Factory function to get a model strategy instance."""
    strategy_info = MODEL_STRATEGY_REGISTRY.get(strategy_type)
    if not strategy_info:
        raise ValueError(f"Unknown model strategy: {strategy_type}")
    return strategy_info.strategy_class()
