from enum import Enum
from dataclasses import dataclass
from typing import Type, Dict, Any, Callable

from src.forecast.strategies.interfaces import DataStrategy
from src.forecast.forecast_implementations import (
    SessionDataStrategy,
    StationChargesDataStrategy,
    StationEnergyDataStrategy
)

@dataclass
class DataStrategyInfo:
    name: str
    display_name: str
    description: str
    strategy_class: Type[DataStrategy]
    default_params: Dict[str, Any]  # Used for feature_engineering
    init_params: Dict[str, Any]  # Used for __init__ if needed

class DataStrategyType(Enum):
    SESSION_ENERGY = "session_energy"
    SESSION_DURATION = "session_duration"
    STATION_CHARGES = "station_charges"
    STATION_ENERGY = "station_energy"

# Registry
DATA_STRATEGY_REGISTRY: Dict[DataStrategyType, DataStrategyInfo] = {
    DataStrategyType.SESSION_ENERGY: DataStrategyInfo(
        name=DataStrategyType.SESSION_ENERGY.value,
        display_name="Session Energy Prediction",
        description="Forecast total energy delivered for a session.",
        strategy_class=SessionDataStrategy,
        default_params={}, # for feature_engineering
        init_params={"target_column": "Energy (kWh)"} # for __init__ of SessionDataStrategy
    ),
    DataStrategyType.SESSION_DURATION: DataStrategyInfo(
        name=DataStrategyType.SESSION_DURATION.value,
        display_name="Session Duration Prediction",
        description="Forecast total duration of a charging session.",
        strategy_class=SessionDataStrategy,
        default_params={},
        init_params={"target_column": "Charge Duration (min)"}
    ),
    DataStrategyType.STATION_CHARGES: DataStrategyInfo(
        name=DataStrategyType.STATION_CHARGES.value,
        display_name="Station Charges Count",
        description="Forecast number of simultaneous charges at a station.",
        strategy_class=StationChargesDataStrategy,
        default_params={},
        init_params={}
    ),
    DataStrategyType.STATION_ENERGY: DataStrategyInfo(
        name=DataStrategyType.STATION_ENERGY.value,
        display_name="Station Energy Demand",
        description="Forecast total energy demand for a station.",
        strategy_class=StationEnergyDataStrategy,
        default_params={},
        init_params={}
    ),
}

def get_data_strategy(strategy_type: DataStrategyType) -> DataStrategy:
    """Factory function to get instantiated data strategy."""
    info = DATA_STRATEGY_REGISTRY[strategy_type]
    # Instantiate with init_params if any
    return info.strategy_class(**info.init_params)
