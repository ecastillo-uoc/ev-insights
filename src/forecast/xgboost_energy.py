from src.forecast.generic_forecast import GenericForecast
from src.forecast.strategies.data.session_data import SessionDataStrategy
from src.forecast.strategies.models.xgboost_strategy import XGBoostStrategy

class xgboost_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='energy_supplied'),
            model_strategy=XGBoostStrategy(output_key='energy'),
            **kwargs
        )
