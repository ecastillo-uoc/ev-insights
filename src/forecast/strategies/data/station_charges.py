import pandas as pd
from src.forecast.strategies.interfaces import DataStrategy
from src.utils.date_utils import utc_to_decimal_hours_minutes
from src.forecast.strategies.ts_utils import add_lags, add_timefeat_df

class StationChargesDataStrategy(DataStrategy):
    def check_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is not None:
            if 'plug_in_datetime' in df.columns:
                df = df.dropna(subset=['plug_in_datetime'])
        return df

    def feature_engineering(self, df: pd.DataFrame, custom_params: dict) -> tuple[pd.DataFrame, list, list]:
        feature_columns = []
        target_columns = []
        
        # We work on a copy to avoid side effects if not intended
        # But for performance on large data, performing in-place might be preferred.
        # Here we follow the pattern of returning the modified df.
        
        for feature, filters in custom_params.items():

            # Plugin Duration in minutes
            if feature == 'plug_in_month':
                # Add feature
                if 'plug_in_month' not in df.columns:
                    df['plug_in_month'] = df['plug_in_datetime'].dt.month
                    feature_columns.append('plug_in_month')

            # Week day name
            if feature == 'plug_in_weekday':
                # Add feature
                if 'plug_in_weekday' not in df.columns:
                    df['plug_in_weekday'] = df['plug_in_datetime'].dt.weekday
                    feature_columns.append('plug_in_weekday')

            # Plugin hour minutes
            if feature == 'plug_in_hour_minutes':
                # Add feature
                if 'plug_in_hour_minutes' not in df.columns:
                    df['plug_in_hour_minutes'] = df['plug_in_datetime'].apply(
                        lambda row: utc_to_decimal_hours_minutes(row))
                    feature_columns.append('plug_in_hour_minutes')

            # Plugin Duration in minutes
            if feature == 'plug_duration':
                # Add feature
                if 'plug_duration' not in df.columns:
                    df['plug_duration'] = (df['plug_out_datetime'] - df[
                        'plug_in_datetime']).dt.total_seconds() / 60
                    feature_columns.append('plug_duration')
                    target_columns.append('plug_duration')

            # Average Duration Moving Averages
            if feature == 'avg_duration':
                for interval in filters:
                    cname = 'avg_duration' + str(interval)
                    if cname not in df.columns:
                        df[cname] = df['plug_duration'].shift(1).rolling(window=interval).mean()
                        feature_columns.append(cname)

            # Average Energy Moving Averages
            if feature == 'avg_energy':
                for interval in filters:
                    cname = 'avg_energy' + str(interval)
                    if cname not in df.columns:
                        df[cname] = df['energy_supplied'].shift(1).rolling(window=interval).mean()
                        feature_columns.append(cname)

            if feature == 'ts_engineering':
                lags = filters['lags']
                lag_windows = filters['lag_windows']

                # Set 'plug_in_datetime' as the index, resample to get daily demand
                # Note: modifying df in place or reassigning
                df.set_index('plug_in_datetime', inplace=True)
                
                # Careful: calling mode() on empty series if filtered too much
                dataset_name_mode = df['dataset_name'].mode()[0]
                station_id_mode = df['station_id'].mode()[0]
                
                df['number_charges'] = df['station_id']
                df = df.resample('D').agg({
                    'number_charges': 'size',
                    'dataset_name': lambda x: dataset_name_mode,
                    'station_id': lambda x: station_id_mode
                })
                
                date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                df = df.reindex(date_range, fill_value=0)
                
                # Use shared utils
                df = add_timefeat_df(df)
                df = add_lags(df, "number_charges", lags, lag_windows)
                
                # Update columns via list comprehension as in original code
                feature_columns = [col for col in df.columns if col not in ['dataset_name', 'date', 'number_charges', 'year']]
                target_columns.append('number_charges')
                
        return df, feature_columns, target_columns
