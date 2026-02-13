from src.forecast.forecast import Forecast
from src.forecast.strategies.interfaces import PredictionTargetStrategy, ModelStrategy

class GenericForecast(Forecast):
    def __init__(self, data_strategy: PredictionTargetStrategy, model_strategy: ModelStrategy, **kwargs):
        super().__init__(**kwargs)
        self.data_strategy = data_strategy
        self.model_strategy = model_strategy
        self.feature_columns = []
        self.target_columns = []
        
    def check_data(self):
        self.df = self.data_strategy.check_data(self.df)

    def feature_engineering(self):
        self.df, self.feature_columns, self.target_columns = self.data_strategy.feature_engineering(
            self.df, self.custom_params
        )
        self.columns = self.feature_columns # Maintain compatibility

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
        # We need the models to predict. 
        # In current Forecast flow, 'self.model' attribute usage is inconsistent (sometimes list, sometimes dict).
        # We can pass the results['train'] or rely on strategy to load models.
        # For this example, we pass self.model (assuming it's loaded in load_data or trained).
        
        output = self.model_strategy.predict(
            df=self.df,
            feature_columns=self.feature_columns,
            target_column=target_col,
            model_objects=self.model, # Strategy needs to know how to handle this
            context_date=self.date
        )
        self.results.update(output)

    def run(self):
        if self.mode == "train":
            self.train()
        elif self.mode == "predict":
            self.predict()
        else:
            self.train()
            self.predict()
