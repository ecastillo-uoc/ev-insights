from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.forecast.strategies.interfaces import PredictionTargetStrategy, ModelStrategy
from src.forecast.strategies.utils_ts import add_lags, add_timefeat_df, smape
from src.utils.date_utils import utc_to_decimal_hours_minutes
from src.forecast.forecast import Forecast

# --- Data Strategies ---

class SessionDataStrategy(PredictionTargetStrategy):
    """
    Data strategy for session-based prediction (Energy or Charge Duration).
    """
    
    def __init__(self, target_column: str):
        self.target_column = target_column

    def feature_engineering(self, df: pd.DataFrame, custom_params: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        columns = []
        target = [self.target_column]
        
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

    def check_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class StationChargesDataStrategy(PredictionTargetStrategy):
    def check_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is not None:
            if 'plug_in_datetime' in df.columns:
                df = df.dropna(subset=['plug_in_datetime'])
        return df

    def feature_engineering(self, df: pd.DataFrame, custom_params: dict) -> tuple[pd.DataFrame, list, list]:
        feature_columns = []
        target_columns = []
        
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
                
                df = add_timefeat_df(df)
                df = add_lags(df, "number_charges", lags, lag_windows)
                
                feature_columns = [col for col in df.columns if col not in ['dataset_name', 'date', 'number_charges', 'year']]
                target_columns.append('number_charges')
                
        return df, feature_columns, target_columns


class StationEnergyDataStrategy(PredictionTargetStrategy):
    """
    Data strategy for aggregated daily station energy demand.
    """
    
    def feature_engineering(self, df: pd.DataFrame, custom_params: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        columns = []
        target = []
        
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

                if 'plug_in_datetime' in df.columns: 
                    df.set_index('plug_in_datetime', inplace=True)
                
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
                
                if not df.empty:
                    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                    df = df.reindex(date_range, fill_value=0)
                
                df = add_timefeat_df(df)
                df = add_lags(df, "daily_demand", lags, lag_windows)
                
                excluded = ['dataset_name', 'date', 'daily_demand', 'year']
                columns = [col for col in df.columns if col not in excluded]
                
                if 'daily_demand' not in target:
                    target.append('daily_demand')

        return df, columns, target

    def check_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


# --- Model Strategies ---

class LightGBMModelStrategy(ModelStrategy):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def train(self, df, feature_columns, target_column, dataset_names, model_name_prefix):
        output_dict = {'train': {}}
        params = {'random_state': 16, 'test_size': 0.20}
        
        for dataset_name in dataset_names:
            subset_df = df.loc[df['dataset_name'] == dataset_name]
            
            X = subset_df[feature_columns]
            y = subset_df[target_column]
            
            n_rows = len(subset_df)
            train_size = int(n_rows * 0.9)
            
            X_train, X_test = X.iloc[:train_size, :], X.iloc[train_size:, :]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            lgb_params = {
                'num_leaves': 10,
                'learning_rate': 0.02,
                'max_depth': 5,
                'verbose': 0,
                'early_stopping_rounds': 200,
                'nthread': -1
            }
            
            lgbtrain = lgb.Dataset(data=X_train, label=y_train, feature_name=feature_columns)
            lgbtest = lgb.Dataset(data=X_test, label=y_test, reference=lgbtrain, feature_name=feature_columns)
            
            lgbm_m = lgb.train(
                lgb_params,
                lgbtrain,
                valid_sets=[lgbtrain, lgbtest],
                callbacks=[lgb.early_stopping(lgb_params['early_stopping_rounds'])]
            )
            
            y_pred_test = lgbm_m.predict(X_test, num_iteration=lgbm_m.best_iteration)
            
            errors = abs(y_pred_test - y_test)
            
            non_zero_indices = y_test != 0
            filtered_errors = errors[non_zero_indices]
            filtered_test_labels = y_test[non_zero_indices]
            
            if len(filtered_test_labels) > 0:
                mape = 100 * np.mean(np.abs(filtered_errors / filtered_test_labels))
            else:
                mape = 0
                
            accuracy = 100 - np.mean(mape)
            smape_val = smape(np.expm1(y_pred_test), np.expm1(y_test)) 
            
            model_name = Forecast.get_model_name(prefix=model_name_prefix, pilot=dataset_name)
            
            output_dict['train'].update({
                model_name: {
                    'params': params,
                    'model': lgbm_m,
                    'metrics': {
                        'smape': round(smape_val, 2),
                        'mape': round(mape, 2),
                        'accuracy': round(accuracy, 2)
                    }
                }
            })
            
        return output_dict

    def predict(self, df, feature_columns, target_column, model_objects, context_date):
        return {'predict': {}}


class XGBoostModelStrategy(ModelStrategy):
    def __init__(self, output_key: str = 'prediction'):
        self.output_key = output_key

    def train(self, 
              df: pd.DataFrame, 
              columns: List[str], 
              target_name: str, 
              dataset_names: List[str], 
              logger: Any,
              model_name_prefix: str) -> Dict[str, Any]:
        
        output_dict = {'train': {}}
        params = {'random_state': 16, 'test_size': 0.20}

        for dataset_name in dataset_names:
            logger.info(f"{dataset_name} - XGBoost Forecast ({target_name}) - Model training")
            
            subset_df = df.loc[df['dataset_name'] == dataset_name].copy()
            X = subset_df[columns].copy() if columns else subset_df.copy()
            
            if 'plug_in_weekday' in X.columns:
                weekday_series = X['plug_in_weekday']
                dums = pd.get_dummies(weekday_series, prefix='plug_in_weekday')
                weekday_cols = [f'plug_in_weekday_{i}' for i in range(7)]
                dums = dums.reindex(columns=weekday_cols, fill_value=0)
                X = X.drop('plug_in_weekday', axis=1)
                X = X.join(dums)
            
            if target_name in X.columns:
                X = X.drop(target_name, axis=1)

            if target_name not in subset_df.columns:
                logger.error(f"Target column {target_name} not found in dataframe")
                continue

            y = subset_df[target_name].astype(float)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=params['test_size'], random_state=params['random_state']
            )
            
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=params['random_state'])
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            errors = abs(preds - y_test)
            mae = np.mean(errors)
            
            non_zero = y_test != 0
            if np.any(non_zero):
                mape = 100 * np.mean(np.abs(errors[non_zero] / y_test[non_zero]))
                accuracy = 100 - np.mean(mape)
            else:
                accuracy = 100
            
            logger.info(f'Mean Absolute Error: {round(mae, 2)}')
            logger.info(f'Accuracy: {round(accuracy, 2)} %.')
            
            pilot_name = f"{model_name_prefix}_{dataset_name}"
            
            output_dict['train'][pilot_name] = {
                'params': params,
                'shapes': {
                    'training_features_shape': X_train.shape,
                    'training_labels_shape': y_train.shape,
                    'testing_features_shape': X_test.shape,
                    'testing_labels_shape': y_test.shape,
                },
                'model': model,
                'metrics': {
                    'mae': round(mae, 2),
                    'accuracy': round(accuracy, 2),
                },
                'artifacts': {}
            }
            
        return output_dict

    def predict(self, 
                model: Any, 
                df: pd.DataFrame, 
                columns: List[str],
                submode: str, 
                dataset_names: List[str]) -> Dict[str, Any]:
        
        output_dict = {'predict': {}}
        
        for dataset_name in dataset_names:
            if submode == 'schedule':
                subset = df.loc[df['dataset_name'] == dataset_name]
                if subset.empty:
                    continue
                
                input_row = subset.iloc[[-1]].copy()
                features = input_row[columns].copy() if columns else input_row.copy()
                
                if 'plug_in_weekday' in features.columns:
                     weekday_val = features['plug_in_weekday'].iloc[0]
                     features = features.drop('plug_in_weekday', axis=1)
                     for i in range(7):
                         features[f'plug_in_weekday_{i}'] = 1 if i == weekday_val else 0

                prediction = model.predict(features)
                val = float(prediction[0])
                
                output_dict['predict'].update({
                    self.output_key: val,
                    'value': val,
                    'date': datetime.now(),
                    'created_at': datetime.now()
                })
        return output_dict

# --- Forecast Definitions ---

from src.forecast.generic_forecast import GenericForecast

class xgboost_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='energy_supplied'),
            model_strategy=XGBoostModelStrategy(output_key='energy'),
            **kwargs
        )

class xgboost_charge_duration(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=SessionDataStrategy(target_column='plug_duration'),
            model_strategy=XGBoostModelStrategy(output_key='duration'),
            **kwargs
        )

class lightgbm_station_charges(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationChargesDataStrategy(),
            model_strategy=LightGBMModelStrategy(),
            **kwargs
        )

class lightgbm_station_energy(GenericForecast):
    def __init__(self, **kwargs):
        super().__init__(
            data_strategy=StationEnergyDataStrategy(),
            model_strategy=LightGBMModelStrategy(),
            **kwargs
        )
