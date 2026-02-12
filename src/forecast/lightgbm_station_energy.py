from src.forecast.generic_forecast import GenericForecast
from src.forecast.strategies.data.station_energy import StationEnergyDataStrategy
from src.forecast.strategies.models.lightgbm_strategy import LightGBMStrategy

class lightgbm_station_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationEnergyDataStrategy(),
            model_strategy=LightGBMStrategy(),
            **kwargs
        )

