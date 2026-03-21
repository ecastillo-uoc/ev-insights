from src.forecast.forecast import Forecast
from src.forecast.strategies.interfaces import PredictionTargetStrategy, ModelStrategy

import logging

class GenericForecast(Forecast):
    def __init__(self, data_strategy: PredictionTargetStrategy, model_strategy: ModelStrategy, **kwargs):
        super().__init__(**kwargs)
        self.data_strategy = data_strategy
        self.model_strategy = model_strategy
        self.feature_columns = []
        self.target_columns = []
        self._gf_logger = logging.getLogger('generic_forecast')
        
    def check_data(self):
        self.df = self.data_strategy.check_data(self.df)

    def feature_engineering(self):
        self.df, self.feature_columns, self.target_columns = self.data_strategy.feature_engineering(
            self.df, self.custom_params
        )
        self.columns = self.feature_columns # Maintain compatibility
        self._gf_logger.info(f"DEBUG [GenericForecast.feature_engineering] "
                             f"df.shape={self.df.shape}, "
                             f"feature_columns={self.feature_columns}, "
                             f"target_columns={self.target_columns}")

    def train(self):
        # Assuming single target for now or handling list inside strategy
        target_col = self.target_columns[0] if self.target_columns else None
        
        output = self.model_strategy.train(
            df=self.df, 
            feature_columns=self.feature_columns, 
            target_column=target_col,
            dataset_names=self.datasets_names,
            model_name_prefix=self.name
        )
        self.results.update(output)

    def predict(self):
        target_col = self.target_columns[0] if self.target_columns else None
        
        self._gf_logger.info(f"DEBUG [GenericForecast.predict] mode={self.mode}, submode={self.submode}, "
                             f"algo={self.algo}, name={self.name}")
        self._gf_logger.info(f"DEBUG [GenericForecast.predict] df.shape={self.df.shape if self.df is not None else None}, "
                             f"feature_columns={self.feature_columns}, "
                             f"target_column={target_col}, "
                             f"datasets_names={list(self.datasets_names)}, "
                             f"model type={type(self.model).__name__}, "
                             f"model is None={self.model is None}")
        
        output = self.model_strategy.predict(
            df=self.df,
            feature_columns=self.feature_columns,
            target_column=target_col,
            model_objects=self.model,
            context_date=self.date,
            dataset_names=self.datasets_names,
            submode=self.submode
        )
        
        self._gf_logger.info(f"DEBUG [GenericForecast.predict] Strategy returned: {list(output.keys()) if output else 'None'}, "
                             f"predict keys: {list(output.get('predict', {}).keys()) if output else 'N/A'}")
        self.results.update(output)

    def run(self):
        if self.mode == "train":
            self.train()
        elif self.mode == "predict":
            self.predict()
        else:
            self.train()
            self.predict()
