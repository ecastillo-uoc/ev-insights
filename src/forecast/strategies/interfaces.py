from abc import ABC, abstractmethod
import pandas as pd

class PredictionTargetStrategy(ABC):
    @abstractmethod
    def feature_engineering(self, df: pd.DataFrame, custom_params: dict) -> tuple[pd.DataFrame, list, list]:
        """
        Perform feature engineering on the dataframe.
        
        Args:
            df: Input dataframe
            custom_params: Dictionary of parameters for feature engineering
            
        Returns:
            Tuple of (processed_dataframe, feature_columns_list, target_columns_list)
        """
        pass

    @abstractmethod
    def check_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and cleans data.
        """
        pass

class ModelStrategy(ABC):
    @abstractmethod
    def train(self, df: pd.DataFrame, feature_columns: list, target_column: str, 
              dataset_names: list, model_name_prefix: str) -> dict:
        """
        Train the model.
        
        Returns:
            Dictionary containing training results/metrics
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame, feature_columns: list, target_column: str, 
                model_objects: dict, context_date) -> dict:
        """
        Make predictions.
        
        Returns:
            Dictionary containing prediction results
        """
        pass
