"""Pre-wired :class:`GenericForecast` subclasses.

Each class pairs a fixed :class:`PredictionTargetStrategy` with a fixed
:class:`ModelStrategy`, providing a convenient configuration layer for the
``init_forecast`` factory when registry-based dynamic composition is not used.
"""

# --- Forecast Definitions ---


from .generic_forecast import GenericForecast
from .strategies import (
    SessionDataStrategy,
    XGBoostModelStrategy,
    StationChargesDataStrategy,
    LightGBMModelStrategy,
    StationEnergyDataStrategy,
    LSTMModelStrategy,
)



class xgboost_session_duration(GenericForecast):
    """XGBoost session-duration forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='plug_duration'),
            model_strategy=XGBoostModelStrategy(output_key='duration'),
            **kwargs
        )


class xgboost_session_energy(GenericForecast):
    """XGBoost session-energy forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='energy_supplied'),
            model_strategy=XGBoostModelStrategy(output_key='energy'),
            **kwargs
        )

class xgboost_station_charges(GenericForecast):
    """XGBoost station-charges forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationChargesDataStrategy(),
            model_strategy=XGBoostModelStrategy(output_key='charges'),
            **kwargs
        )

class xgboost_station_energy(GenericForecast):
    """XGBoost station-energy forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationEnergyDataStrategy(),
            model_strategy=XGBoostModelStrategy(output_key='energy'),
            **kwargs
        )



class lightgbm_station_charges(GenericForecast):
    """LightGBM station-charges forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationChargesDataStrategy(),
            model_strategy=LightGBMModelStrategy(),
            **kwargs
        )

class lightgbm_station_energy(GenericForecast):
    """LightGBM station-energy forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationEnergyDataStrategy(),
            model_strategy=LightGBMModelStrategy(),
            **kwargs
        )

class lightgbm_session_energy(GenericForecast):
    """LightGBM session-energy forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='energy_supplied'),
            model_strategy=LightGBMModelStrategy(),
            **kwargs
        )

class lightgbm_session_duration(GenericForecast):
    """LightGBM session-duration forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='plug_duration'),
            model_strategy=LightGBMModelStrategy(),
            **kwargs
        )





class lstm_station_energy(GenericForecast):
    """LSTM station-energy forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationEnergyDataStrategy(),
            model_strategy=LSTMModelStrategy(output_key='energy'),
            **kwargs
        )


class lstm_session_energy(GenericForecast):
    """LSTM session-energy forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='energy_supplied'),
            model_strategy=LSTMModelStrategy(output_key='energy'),
            **kwargs
        )

class lstm_station_charges(GenericForecast):
    """LSTM station-charges forecaster."""
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationChargesDataStrategy(),
            model_strategy=LSTMModelStrategy(output_key='charges'),
            **kwargs
        )
