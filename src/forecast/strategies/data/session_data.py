from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from src.forecast.strategies.interfaces import DataStrategy
from src.utils.date_utils import utc_to_decimal_hours_minutes

class SessionDataStrategy(DataStrategy):
    """
    Data strategy for session-based prediction (Energy or Charge Duration).
    Corresponds to logic in xgboost_energy.py and xgboost_charge_duration.py
    """
    
    def __init__(self, target_column: str):
        self.target_column = target_column

    def feature_engineering(self, df: pd.DataFrame, custom_params: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        columns = []
        target = [self.target_column]
        
        # Consistent with xgboost_energy.py:
        # self.columns.append('energy_supplied') <-- wait, xgboost_energy appends target to columns initially?
        # Actually in xgboost_energy.py line 32: self.columns.append('energy_supplied')
        # But later in train(): target = np.array(df['energy_supplied']); df = df.drop('energy_supplied', axis=1)
        # So we should probably NOT include target in columns list for training features, 
        # but ensure it exists in DF.
        
        # Note: GenericForecast expects 'columns' to be the feature columns used for X.
        
        for feature, filters in custom_params.items():

            # Plugin month
            if feature == 'plug_in_month':
                if 'plug_in_month' not in df.columns:
                    df['plug_in_month'] = df['plug_in_datetime'].dt.month
                    columns.append('plug_in_month')

            # Week day name
            if feature == 'plug_in_weekday':
                if 'plug_in_weekday' not in df.columns:
                    df['plug_in_weekday'] = df['plug_in_datetime'].dt.weekday
                    columns.append('plug_in_weekday')

            # Plugin hour minutes
            if feature == 'plug_in_hour_minutes':
                if 'plug_in_hour_minutes' not in df.columns:
                    df['plug_in_hour_minutes'] = df['plug_in_datetime'].apply(
                        lambda row: utc_to_decimal_hours_minutes(row))
                    columns.append('plug_in_hour_minutes')

            # Plugin Duration in minutes
            if feature == 'plug_duration':
                # Re-calculate or ensure it exists
                if 'plug_duration' not in df.columns:
                     df['plug_duration'] = (df['plug_out_datetime'] - df['plug_in_datetime']).dt.total_seconds() / 60
                
                # If target is plug_duration, we don't add it to features usually, 
                # but xgboost_charge_duration.py line 49 adds it to columns if feature=='plug_duration'. 
                # Wait, xgboost_charge_duration.py uses 'plug_duration' as target.
                # If 'plug_duration' is the target, we shouldn't add it to input features (leakage) unless for lag calculation?
                # The original code adds it to columns if requested. 
                # But later in train(): target = np.array(df['plug_duration']); df = df.drop('plug_duration', axis=1)
                # So it removes it.
                
                if feature == 'plug_duration' and self.target_column != 'plug_duration':
                     columns.append('plug_duration')

            # Average Duration Moving Averages
            if feature == 'avg_duration':
                for interval in filters:
                    cname = 'avg_duration' + str(interval)
                    if cname not in df.columns:
                        df[cname] = df['plug_duration'].shift(1).rolling(window=interval).mean()
                        columns.append(cname)

            # Average Energy Moving Averages
            if feature == 'avg_energy':
                for interval in filters:
                    cname = 'avg_energy' + str(interval)
                    if cname not in df.columns:
                        df[cname] = df['energy_supplied'].shift(1).rolling(window=interval).mean()
                        columns.append(cname)

        return df, columns, target

    def check_data(self, df: pd.DataFrame) -> None:
        # Example check
        pass
