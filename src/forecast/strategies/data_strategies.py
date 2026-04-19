"""
Data strategies for EV charging session and station-level prediction targets.

Each strategy implements the ``PredictionTargetStrategy`` interface, providing:
  * ``feature_engineering()`` — derives model-ready features from raw session data
  * ``check_data()`` — validates / cleans the incoming dataframe

Three strategies are provided:

``SessionDataStrategy``
    Per-session features (month, weekday, hour, duration, rolling averages).
    Parameterised by *target_column* (``'energy_supplied'`` or ``'plug_duration'``).

``StationChargesDataStrategy``
    Daily charge-count aggregation with lag and time-feature engineering.

``StationEnergyDataStrategy``
    Daily energy-demand aggregation (``energy_supplied`` summed per day)
    with lag and time-feature engineering.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.utils.date_utils import utc_to_decimal_hours_minutes
from src.data.constants import COVID_START, COVID_END, EXCLUDE_COVID_DATA
from .interfaces import PredictionTargetStrategy
from .utils_ts import add_lags, add_timefeat_df


class SessionDataStrategy(PredictionTargetStrategy):
    """Per-session feature engineering for energy or duration prediction."""

    def __init__(self, target_column: str):
        self.target_column = target_column

    def feature_engineering(
        self, df: pd.DataFrame, custom_params: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        columns: List[str] = []
        target = [self.target_column]

        for feature, filters in custom_params.items():

            if feature == 'plug_in_month':
                if 'plug_in_month' not in df.columns:
                    df['plug_in_month'] = df['plug_in_datetime'].dt.month
                    columns.append('plug_in_month')

            if feature == 'plug_in_weekday':
                if 'plug_in_weekday' not in df.columns:
                    df['plug_in_weekday'] = df['plug_in_datetime'].dt.weekday
                    columns.append('plug_in_weekday')

            if feature == 'plug_in_hour_minutes':
                if 'plug_in_hour_minutes' not in df.columns:
                    df['plug_in_hour_minutes'] = df['plug_in_datetime'].apply(
                        lambda row: utc_to_decimal_hours_minutes(row))
                    columns.append('plug_in_hour_minutes')

            if feature == 'plug_duration':
                if 'plug_duration' not in df.columns:
                    df['plug_duration'] = (
                        df['plug_out_datetime'] - df['plug_in_datetime']
                    ).dt.total_seconds() / 60

                if feature == 'plug_duration' and self.target_column != 'plug_duration':
                    columns.append('plug_duration')

            if feature == 'avg_duration':
                for interval in filters:
                    cname = 'avg_duration' + str(interval)
                    if cname not in df.columns:
                        df[cname] = df['plug_duration'].shift(1).rolling(window=interval).mean()
                        columns.append(cname)

            if feature == 'avg_energy':
                for interval in filters:
                    cname = 'avg_energy' + str(interval)
                    if cname not in df.columns:
                        df[cname] = df['energy_supplied'].shift(1).rolling(window=interval).mean()
                        columns.append(cname)

        return df, columns, target

    def check_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class StationChargesDataStrategy(PredictionTargetStrategy):
    """Daily charge-count aggregation with time-feature and lag engineering."""

    def check_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is not None:
            if 'plug_in_datetime' in df.columns:
                df = df.dropna(subset=['plug_in_datetime'])
        return df

    def feature_engineering(
        self, df: pd.DataFrame, custom_params: dict,
    ) -> tuple[pd.DataFrame, list, list]:
        feature_columns: list = []
        target_columns: list = []

        for feature, filters in custom_params.items():

            if feature == 'plug_in_month':
                if 'plug_in_month' not in df.columns:
                    df['plug_in_month'] = df['plug_in_datetime'].dt.month
                    feature_columns.append('plug_in_month')

            if feature == 'plug_in_weekday':
                if 'plug_in_weekday' not in df.columns:
                    df['plug_in_weekday'] = df['plug_in_datetime'].dt.weekday
                    feature_columns.append('plug_in_weekday')

            if feature == 'plug_in_hour_minutes':
                if 'plug_in_hour_minutes' not in df.columns:
                    df['plug_in_hour_minutes'] = df['plug_in_datetime'].apply(
                        lambda row: utc_to_decimal_hours_minutes(row))
                    feature_columns.append('plug_in_hour_minutes')

            if feature == 'plug_duration':
                if 'plug_duration' not in df.columns:
                    df['plug_duration'] = (
                        df['plug_out_datetime'] - df['plug_in_datetime']
                    ).dt.total_seconds() / 60
                    feature_columns.append('plug_duration')
                    target_columns.append('plug_duration')

            if feature == 'avg_duration':
                for interval in filters:
                    cname = 'avg_duration' + str(interval)
                    if cname not in df.columns:
                        df[cname] = df['plug_duration'].shift(1).rolling(window=interval).mean()
                        feature_columns.append(cname)

            if feature == 'avg_energy':
                for interval in filters:
                    cname = 'avg_energy' + str(interval)
                    if cname not in df.columns:
                        df[cname] = df['energy_supplied'].shift(1).rolling(window=interval).mean()
                        feature_columns.append(cname)

            if feature == 'ts_engineering':
                lags = filters['lags']
                lag_windows = filters['lag_windows']

                if 'plug_in_datetime' in df.columns:
                    df.set_index('plug_in_datetime', inplace=True)

                dataset_name_mode = df['dataset_name'].mode()[0] if not df['dataset_name'].empty else 'unknown'
                station_id_mode = df['station_id'].mode()[0] if not df['station_id'].empty else 'unknown'

                df['number_charges'] = df['station_id']
                df = df.resample('D').agg({
                    'number_charges': 'size',
                    'dataset_name': lambda x: dataset_name_mode,
                    'station_id': lambda x: station_id_mode
                })

                if not df.empty:
                    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                    df = df.reindex(date_range, fill_value=0)

                # Remove COVID no-data period (artificial zeros from reindex)
                if EXCLUDE_COVID_DATA:
                    covid_mask = (df.index >= pd.to_datetime(COVID_START)) & (df.index <= pd.to_datetime(COVID_END))
                    df = df[~covid_mask]

                df = add_timefeat_df(df)
                df = add_lags(df, "number_charges", lags, lag_windows)

                feature_columns = [
                    col for col in df.columns
                    if col not in ['dataset_name', 'date', 'number_charges', 'year']
                ]
                target_columns.append('number_charges')

        return df, feature_columns, target_columns


class StationEnergyDataStrategy(PredictionTargetStrategy):
    """Daily energy-demand aggregation with time-feature and lag engineering."""

    def feature_engineering(
        self, df: pd.DataFrame, custom_params: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        columns: List[str] = []
        target: List[str] = []

        for feature, filters in custom_params.items():

            if feature == 'plug_in_month':
                if 'plug_in_month' not in df.columns:
                    df['plug_in_month'] = df['plug_in_datetime'].dt.month
                    columns.append('plug_in_month')

            if feature == 'plug_in_weekday':
                if 'plug_in_weekday' not in df.columns:
                    df['plug_in_weekday'] = df['plug_in_datetime'].dt.weekday
                    columns.append('plug_in_weekday')

            if feature == 'plug_in_hour_minutes':
                if 'plug_in_hour_minutes' not in df.columns:
                    df['plug_in_hour_minutes'] = df['plug_in_datetime'].apply(
                        lambda row: utc_to_decimal_hours_minutes(row))
                    columns.append('plug_in_hour_minutes')

            if feature == 'plug_duration':
                if 'plug_duration' not in df.columns:
                    df['plug_duration'] = (
                        df['plug_out_datetime'] - df['plug_in_datetime']
                    ).dt.total_seconds() / 60
                    columns.append('plug_duration')

            if feature == 'avg_duration':
                for interval in filters:
                    cname = 'avg_duration' + str(interval)
                    if cname not in df.columns:
                        df[cname] = df['plug_duration'].shift(1).rolling(window=interval).mean()
                        columns.append(cname)

            if feature == 'avg_energy':
                for interval in filters:
                    cname = 'avg_energy' + str(interval)
                    if cname not in df.columns:
                        df[cname] = df['energy_supplied'].shift(1).rolling(window=interval).mean()
                        columns.append(cname)

            if feature == 'ts_engineering':
                lags = filters['lags']
                lag_windows = filters['lag_windows']

                if 'plug_in_datetime' in df.columns:
                    df.set_index('plug_in_datetime', inplace=True)

                dataset_name_mode = df['dataset_name'].mode()[0] if not df['dataset_name'].empty else 'unknown'
                station_id_mode = df['station_id'].mode()[0] if not df['station_id'].empty else 'unknown'

                df = df.resample('D').agg({
                    'energy_supplied': 'sum',
                    'dataset_name': lambda x: dataset_name_mode,
                    'station_id': lambda x: station_id_mode
                })

                df.rename(columns={'energy_supplied': 'daily_demand'}, inplace=True)
                df['daily_demand'] = pd.to_numeric(df['daily_demand'], errors='coerce')

                if not df.empty:
                    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                    df = df.reindex(date_range, fill_value=0)

                # Remove COVID no-data period (artificial zeros from reindex)
                if EXCLUDE_COVID_DATA:
                    covid_mask = (df.index >= pd.to_datetime(COVID_START)) & (df.index <= pd.to_datetime(COVID_END))
                    df = df[~covid_mask]

                df = add_timefeat_df(df)
                df = add_lags(df, "daily_demand", lags, lag_windows)

                excluded = ['dataset_name', 'date', 'daily_demand', 'year']
                columns = [col for col in df.columns if col not in excluded]

                if 'daily_demand' not in target:
                    target.append('daily_demand')

        return df, columns, target

    def check_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
