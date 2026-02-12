from src.forecast.generic_forecast import GenericForecast
from src.forecast.strategies.data.station_charges import StationChargesDataStrategy
from src.forecast.strategies.models.lightgbm_strategy import LightGBMStrategy

class lightgbm_station_charges(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationChargesDataStrategy(),
            model_strategy=LightGBMStrategy(),
            **kwargs
        )

