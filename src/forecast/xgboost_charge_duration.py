from src.forecast.generic_forecast import GenericForecast
from src.forecast.strategies.data.session_data import SessionDataStrategy
from src.forecast.strategies.models.xgboost_strategy import XGBoostStrategy

class xgboost_charge_duration(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='plug_duration'),
            model_strategy=XGBoostStrategy(output_key='duration'),
            **kwargs
        )

