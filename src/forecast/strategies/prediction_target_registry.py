from enum import Enum
from dataclasses import dataclass
from typing import Type, Dict, Any, Callable

from src.forecast.strategies.interfaces import PredictionTargetStrategy
from src.forecast.forecast_implementations import (
    SessionDataStrategy,
    StationChargesDataStrategy,
    StationEnergyDataStrategy
)

@dataclass
class PredictionTargetInfo:
    name: str # Internal ID
    display_name: str # UI Label
    description: str # UI Tooltip
    strategy_class: Type[PredictionTargetStrategy] # Implementation class
    default_params: Dict[str, Any]  # Used for feature_engineering
    init_params: Dict[str, Any]  # Used for __init__ of the strategy

class PredictionTarget(Enum):
    SESSION_ENERGY = "session_energy"
    SESSION_DURATION = "session_duration"
    STATION_CHARGES = "station_charges"
    STATION_ENERGY = "station_energy"

# Registry
PREDICTION_TARGET_REGISTRY: Dict[PredictionTarget, PredictionTargetInfo] = {
    PredictionTarget.SESSION_ENERGY: PredictionTargetInfo(
        name=PredictionTarget.SESSION_ENERGY.value,
        display_name="Session Energy Prediction",
        description="Forecast total energy delivered for a session.",
        strategy_class=SessionDataStrategy,
        default_params={}, 
        init_params={"target_column": "energy_supplied"}
    ),
    PredictionTarget.SESSION_DURATION: PredictionTargetInfo(
        name=PredictionTarget.SESSION_DURATION.value,
        display_name="Session Duration Prediction",
        description="Forecast total duration of a charging session.",
        strategy_class=SessionDataStrategy,
        default_params={},
        init_params={"target_column": "plug_duration"}
    ),
    PredictionTarget.STATION_CHARGES: PredictionTargetInfo(
        name=PredictionTarget.STATION_CHARGES.value,
        display_name="Station Charges Count",
        description="Forecast number of simultaneous charges at a station.",
        strategy_class=StationChargesDataStrategy,
        default_params={},
        init_params={}
    ),
    PredictionTarget.STATION_ENERGY: PredictionTargetInfo(
        name=PredictionTarget.STATION_ENERGY.value,
        display_name="Station Energy Demand",
        description="Forecast total energy demand for a station.",
        strategy_class=StationEnergyDataStrategy,
        default_params={},
        init_params={}
    ),
}

def get_prediction_target_strategy(target_type: PredictionTarget) -> PredictionTargetStrategy:
    """Factory function to get instantiated strategy for a target."""
    info = PREDICTION_TARGET_REGISTRY[target_type]
    return info.strategy_class(**info.init_params)
