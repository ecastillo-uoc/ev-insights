from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from src.forecast.strategies.interfaces import DataStrategy
from src.utils.date_utils import utc_to_decimal_hours_minutes
from src.forecast.strategies.ts_utils import add_lags, add_timefeat_df

class StationEnergyDataStrategy(DataStrategy):
    """
    Data strategy for aggregated daily station energy demand.
    Corresponds to logic in lightgbm_station_energy.py
    """
    
    def feature_engineering(self, df: pd.DataFrame, custom_params: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        columns = []
        target = []
        
        # Initial columns from df that we might want to keep or use as base
        # But looking at original code, it builds columns list incrementally
        
        for feature, filters in custom_params.items():
            
            # Plugin Duration in minutes
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
                if 'plug_duration' not in df.columns:
                    df['plug_duration'] = (df['plug_out_datetime'] - df['plug_in_datetime']).dt.total_seconds() / 60
                    # Note: original code appends to target here? No, target is usually separate.
                    # In lightgbm_station_energy.py, plug_duration is added to columns, but target is daily_demand.
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

            # Time Series Aggregation
            if feature == 'ts_engineering':
                lags = filters['lags']
                lag_windows = filters['lag_windows']

                # Set 'plug_in_datetime' as the index, resample to get daily demand
                if 'plug_in_datetime' in df.columns: 
                    df.set_index('plug_in_datetime', inplace=True)
                
                # Check if safe to access mode
                dataset_name_mode = df['dataset_name'].mode()[0] if not df['dataset_name'].empty else 'unknown'
                station_id_mode = df['station_id'].mode()[0] if not df['station_id'].empty else 'unknown'
                
                # Resample to Daily
                df = df.resample('D').agg({
                    'energy_supplied': 'sum',
                    'dataset_name': lambda x: dataset_name_mode,
                    'station_id': lambda x: station_id_mode
                })
                
                df.rename(columns={'energy_supplied': 'daily_demand'}, inplace=True)
                df['daily_demand'] = pd.to_numeric(df['daily_demand'], errors='coerce')
                
                # Ensure date range is complete
                if not df.empty:
                    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                    df = df.reindex(date_range, fill_value=0)
                
                df = add_timefeat_df(df)
                df = add_lags(df, "daily_demand", lags, lag_windows)
                
                # Update columns logic - mimicking original
                # "self.columns = [col for col in self.df.columns if col not in ['dataset_name', 'date', 'daily_demand', 'year']]"
                excluded = ['dataset_name', 'date', 'daily_demand', 'year']
                columns = [col for col in df.columns if col not in excluded]
                
                # In this specific strategy, the target is implicitly 'daily_demand'
                if 'daily_demand' not in target:
                    target.append('daily_demand')

        return df, columns, target

    def check_data(self, df: pd.DataFrame) -> None:
        pass
