
# --- Forecast Definitions ---


from .generic_forecast import GenericForecast
from .forecast_implementations import (
    SessionDataStrategy,
    XGBoostModelStrategy,
    StationChargesDataStrategy,
    LightGBMModelStrategy,
    StationEnergyDataStrategy,
    LSTMModelStrategy
)

class xgboost_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='energy_supplied'),
            model_strategy=XGBoostModelStrategy(output_key='energy'),
            **kwargs
        )

class xgboost_charge_duration(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='plug_duration'),
            model_strategy=XGBoostModelStrategy(output_key='duration'),
            **kwargs
        )

class lightgbm_station_charges(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationChargesDataStrategy(),
            model_strategy=LightGBMModelStrategy(),
            **kwargs
        )

class lightgbm_station_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationEnergyDataStrategy(),
            model_strategy=LightGBMModelStrategy(),
            **kwargs
        )

class lstm_station_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationEnergyDataStrategy(),
            model_strategy=LSTMModelStrategy(output_key='energy'),
            **kwargs
        )

# --- Additional forecast class combinations ---

class lightgbm_session_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='energy_supplied'),
            model_strategy=LightGBMModelStrategy(),
            **kwargs
        )

class lightgbm_session_duration(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='plug_duration'),
            model_strategy=LightGBMModelStrategy(),
            **kwargs
        )

class xgboost_session_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='energy_supplied'),
            model_strategy=XGBoostModelStrategy(output_key='energy'),
            **kwargs
        )

class xgboost_station_charges(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationChargesDataStrategy(),
            model_strategy=XGBoostModelStrategy(output_key='charges'),
            **kwargs
        )

class xgboost_station_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationEnergyDataStrategy(),
            model_strategy=XGBoostModelStrategy(output_key='energy'),
            **kwargs
        )

class lstm_session_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='energy_supplied'),
            model_strategy=LSTMModelStrategy(output_key='energy'),
            **kwargs
        )

class lstm_station_charges(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationChargesDataStrategy(),
            model_strategy=LSTMModelStrategy(output_key='charges'),
            **kwargs
        )
