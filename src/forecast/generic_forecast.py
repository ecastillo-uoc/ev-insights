from .forecast import Forecast
from .strategies.interfaces import PredictionTargetStrategy, ModelStrategy

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

    def _resolve_model_strategy(self):
        """Auto-detect model type and swap strategy if there's a mismatch."""
        if self.model is None:
            return
        model_type = type(self.model).__name__
        strategy_type = type(self.model_strategy).__name__

        # Lazy imports to avoid circular dependency at module load time
        if model_type == 'Booster' and strategy_type != 'LightGBMModelStrategy':
            from src.forecast.forecast_implementations import LightGBMModelStrategy
            self._gf_logger.info(f"Auto-switching from {strategy_type} to LightGBMModelStrategy (detected {model_type})")
            self.model_strategy = LightGBMModelStrategy()
        elif model_type in ('XGBRegressor', 'XGBClassifier') and strategy_type != 'XGBoostModelStrategy':
            from src.forecast.forecast_implementations import XGBoostModelStrategy
            self._gf_logger.info(f"Auto-switching from {strategy_type} to XGBoostModelStrategy (detected {model_type})")
            self.model_strategy = XGBoostModelStrategy()
        elif isinstance(self.model, dict) and 'keras_model' in self.model and strategy_type != 'LSTMModelStrategy':
            from src.forecast.forecast_implementations import LSTMModelStrategy
            self._gf_logger.info(f"Auto-switching from {strategy_type} to LSTMModelStrategy (detected keras dict)")
            self.model_strategy = LSTMModelStrategy()

    def predict(self):
        target_col = self.target_columns[0] if self.target_columns else None

        # Auto-detect and correct model strategy mismatch
        self._resolve_model_strategy()
        
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
