from enum import Enum
from dataclasses import dataclass
from typing import Type, Dict, Any, Callable

from .strategies.interfaces import PredictionTargetStrategy
from .strategies import (
    SessionDataStrategy,
    StationChargesDataStrategy,
    StationEnergyDataStrategy,
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

# Default fields required by each prediction target for get_data() SQL query.
# These are merged into data_selection['fields'] when the config leaves them empty.
_SESSION_FIELDS = {
    "plug_in_datetime": None,
    "plug_out_datetime": None,
    "energy_supplied": None,
    "ev_max_charging_power": None,
}

_SESSION_ENERGY_CUSTOM_PARAMS = {
    "plug_in_month": None,
    "plug_in_weekday": None,
    "plug_in_hour_minutes": None,
    "plug_duration": None,
    "avg_energy": [2, 7, 14],
    "avg_duration": [2, 7, 14],
}

_SESSION_DURATION_CUSTOM_PARAMS = {
    "plug_in_month": None,
    "plug_in_weekday": None,
    "plug_in_hour_minutes": None,
    "plug_duration": None,
    "avg_energy": [2, 7, 14],
    "avg_duration": [2, 7, 14],
}

_STATION_FIELDS = {
    "plug_in_datetime": None,
    "plug_out_datetime": None,
    "energy_supplied": None,
    "station_id": None,
}

_STATION_CHARGES_CUSTOM_PARAMS = {
    "ts_engineering": {"lags": [7, 14, 28], "lag_windows": [7, 14, 28]},
}

_STATION_ENERGY_CUSTOM_PARAMS = {
    "ts_engineering": {"lags": [7, 14, 28], "lag_windows": [7, 14, 28]},
}

# Registry
PREDICTION_TARGET_REGISTRY: Dict[PredictionTarget, PredictionTargetInfo] = {
    PredictionTarget.SESSION_ENERGY: PredictionTargetInfo(
        name=PredictionTarget.SESSION_ENERGY.value,
        display_name="Session Energy Prediction",
        description="Forecast total energy delivered for a session.",
        strategy_class=SessionDataStrategy,
        default_params={"fields": _SESSION_FIELDS, "custom_params": _SESSION_ENERGY_CUSTOM_PARAMS},
        init_params={"target_column": "energy_supplied"}
    ),
    PredictionTarget.SESSION_DURATION: PredictionTargetInfo(
        name=PredictionTarget.SESSION_DURATION.value,
        display_name="Session Duration Prediction",
        description="Forecast total duration of a charging session.",
        strategy_class=SessionDataStrategy,
        default_params={"fields": _SESSION_FIELDS, "custom_params": _SESSION_DURATION_CUSTOM_PARAMS},
        init_params={"target_column": "plug_duration"}
    ),
    PredictionTarget.STATION_CHARGES: PredictionTargetInfo(
        name=PredictionTarget.STATION_CHARGES.value,
        display_name="Station Charges Count",
        description="Forecast number of simultaneous charges at a station.",
        strategy_class=StationChargesDataStrategy,
        default_params={"fields": _STATION_FIELDS, "custom_params": _STATION_CHARGES_CUSTOM_PARAMS},
        init_params={}
    ),
    PredictionTarget.STATION_ENERGY: PredictionTargetInfo(
        name=PredictionTarget.STATION_ENERGY.value,
        display_name="Station Energy Demand",
        description="Forecast total energy demand for a station.",
        strategy_class=StationEnergyDataStrategy,
        default_params={"fields": _STATION_FIELDS, "custom_params": _STATION_ENERGY_CUSTOM_PARAMS},
        init_params={}
    ),
}

def get_prediction_target_strategy(target_type: PredictionTarget) -> PredictionTargetStrategy:
    """Factory function to get instantiated strategy for a target."""
    info = PREDICTION_TARGET_REGISTRY[target_type]
    return info.strategy_class(**info.init_params)
