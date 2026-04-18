from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Tree Models
import lightgbm as lgb
import xgboost as xgb

# Keras
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, LSTM as KerasLSTM, Dense, Dropout, Flatten
from keras.layers import MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D, Reshape
from keras.utils import plot_model
# Own modules
from src.utils.console import Colors
from src.utils.date_utils import utc_to_decimal_hours_minutes

from .strategies.interfaces import PredictionTargetStrategy, ModelStrategy
from .strategies.utils_ts import add_lags, add_timefeat_df, smape

#######################################################################################
# --- Data Strategies ---
#######################################################################################
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

#######################################################################################
# --- Model Strategies ---
#######################################################################################
class LightGBMModelStrategy(ModelStrategy):
    def __init__(self, output_key: str = 'prediction'):
        self.output_key = output_key
        self.logger = logging.getLogger(__name__)

    def train(self, df, feature_columns, target_column, dataset_names, model_name_prefix, split_date=None):
        from .forecast import Forecast
        output_dict = {'train': {}}
        params = {'random_state': 16, 'test_size': 0.20}
        
        try:

            for dataset_name in dataset_names:
                subset_df = df.loc[df['dataset_name'] == dataset_name]
                
                if feature_columns:
                    X = subset_df[feature_columns].copy()
                else:
                    X = subset_df.copy()
                    cols_to_drop = [target_column, 'dataset_name', 'date', 'split_date', 'plug_in_datetime']
                    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
                
                y = subset_df[target_column]
                
                if split_date is not None:
                    # Filter by date string assuming daily index or split column
                    if isinstance(subset_df.index, pd.DatetimeIndex):
                        train_mask = subset_df.index <= split_date
                        test_mask = subset_df.index > split_date
                    else:
                        # Fallback if no datetime index
                        n_rows = len(subset_df)
                        train_size = int(n_rows * 0.9)
                        train_mask = np.arange(n_rows) < train_size
                        test_mask = np.arange(n_rows) >= train_size
                    
                    X_train, X_test = X[train_mask], X[test_mask]
                    y_train, y_test = y[train_mask], y[test_mask]
                else:
                    n_rows = len(subset_df)
                    train_size = int(n_rows * 0.9)
                    
                    X_train, X_test = X.iloc[:train_size, :], X.iloc[train_size:, :]
                    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
                
                print(f"X_train shape: {X_train.shape}")
                print(f"X_train dtypes:\n{X_train.dtypes}")
                
                # Exclude object/string features which LightGBM can't handle natively
                X_train = X_train.select_dtypes(exclude=['object', 'string'])
                X_test = X_test.select_dtypes(exclude=['object', 'string'])
                features_to_use = list(X_train.columns)

                lgb_params = {
                    'num_leaves': 10,
                    'learning_rate': 0.02,
                    'max_depth': 5,
                    'verbose': 0,
                    'early_stopping_rounds': 200,
                    'nthread': -1
                }
                
                features_to_use = list(X_train.columns)

                lgbtrain = lgb.Dataset(data=X_train, label=y_train, feature_name=features_to_use)
                lgbtest = lgb.Dataset(data=X_test, label=y_test, reference=lgbtrain, feature_name=features_to_use)
                
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
        except Exception as e:
            self.logger.error(f"LightGBM train failed: {e}\n{traceback.format_exc()}")
            raise

        return output_dict

    def predict(self, df, feature_columns, target_column, model_objects, context_date, dataset_names=None, submode=None):
        output_dict = {'predict': {}}
        X = None
        subset_dates = None
        dates_list = None
        actual_vals = None
        try:
            debug_msg = f"[LightGBM.predict] Starting predict: " + \
                        f"df.shape={df.shape if df is not None else None}, " + \
                        f"feature_columns={feature_columns}, " + \
                        f"target_column={target_column}, " + \
                        f"model_objects type={type(model_objects).__name__}, " + \
                        f"context_date={context_date}, " + \
                        f"dataset_names={dataset_names}, " + \
                        f"submode={submode}"
            self.logger.debug(debug_msg)

            model = model_objects
            if model is None:
                self.logger.error("[LightGBM.predict] model_objects is None – cannot predict")
                return output_dict

            # Use the model's own feature names to ensure predict matches training
            if hasattr(model, 'feature_name'):
                model_features = model.feature_name()
                self.logger.debug(f"[LightGBM.predict] Model trained features ({len(model_features)}): {model_features}")
                missing = [f for f in model_features if f not in df.columns]
                if missing:
                    self.logger.error(f"[LightGBM.predict] Missing features in df: {missing}")
                # Use model's features instead of feature_columns from data strategy
                feature_columns = model_features

            for dataset_name in (dataset_names or []):
                self.logger.debug(f"[LightGBM.predict] Processing dataset: {dataset_name}")
                subset = df.loc[df['dataset_name'] == dataset_name]
                self.logger.debug(f"[LightGBM.predict] Subset rows: {len(subset)}")

                if subset.empty:
                    self.logger.warning(f"[LightGBM.predict] Empty subset for {dataset_name}, skipping")
                    continue

                # Extract dates from index (ts_engineering) or plug_in_datetime column
                if isinstance(subset.index, pd.DatetimeIndex):
                    subset_dates = subset.index
                elif 'plug_in_datetime' in subset.columns:
                    subset_dates = pd.DatetimeIndex(subset['plug_in_datetime'])
                else:
                    subset_dates = None

                if submode == 'schedule':
                    # Take the last row as input for a single-step forecast
                    input_row = subset.iloc[[-1]].copy()
                    X = input_row[feature_columns]
                    self.logger.debug(f"[LightGBM.predict] Schedule mode, input_row features: {X.columns.tolist()}, values: {X.values.tolist()}")
                else:
                    # Use all rows
                    X = subset[feature_columns]
                    self.logger.debug(f"[LightGBM.predict] Full mode, X.shape={X.shape}")

                prediction = model.predict(X, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None)
                self.logger.debug(f"[LightGBM.predict] Raw prediction: {prediction}")

                if submode == 'schedule':
                    val = float(prediction[0])
                    output_dict['predict'].update({
                        'value': val,
                        'date': context_date if context_date else datetime.now(),
                        'created_at': datetime.now()
                    })
                else:
                    dates_list = subset_dates.strftime('%Y-%m-%d').tolist() if subset_dates is not None else None
                    # Include actual values for comparison plotting
                    actual_vals = subset[target_column].tolist() if target_column and target_column in subset.columns else None
                    output_dict['predict'].update({
                        'values': prediction.tolist(),
                        'dates': dates_list,
                        'actuals': actual_vals,
                        'date': context_date if context_date else datetime.now(),
                        'created_at': datetime.now()
                    })

            self.logger.debug(f"[LightGBM.predict] Final output keys: {list(output_dict['predict'].keys())}")
        except Exception as e:
            self.logger.error(f"LightGBM predict failed: {e}\n{traceback.format_exc()}")
            raise

        return output_dict


class XGBoostModelStrategy(ModelStrategy):
    def __init__(self, output_key: str = 'prediction'):
        self.output_key = output_key
        self.logger = logging.getLogger(__name__)

    def train(self,
              df: pd.DataFrame,
              feature_columns: List[str],
              target_column: str,
              dataset_names: List[str],
              model_name_prefix: str,
              split_date: str = None) -> Dict[str, Any]:
        
        output_dict = {'train': {}}
        params = {'random_state': 16, 'test_size': 0.20}

        try:
            for dataset_name in dataset_names:
                self.logger.info(f"{dataset_name} - XGBoost Forecast ({target_column}) - Model training")
                
                subset_df = df.loc[df['dataset_name'] == dataset_name].copy()
                if feature_columns:
                    X = subset_df[feature_columns].copy()
                else:
                    X = subset_df.copy()
                    cols_to_drop = [target_column, 'dataset_name', 'date', 'split_date', 'plug_in_datetime']
                    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
                
                if 'plug_in_weekday' in X.columns:
                    weekday_series = X['plug_in_weekday']
                    dums = pd.get_dummies(weekday_series, prefix='plug_in_weekday')
                    weekday_cols = [f'plug_in_weekday_{i}' for i in range(7)]
                    dums = dums.reindex(columns=weekday_cols, fill_value=0)
                    X = X.drop('plug_in_weekday', axis=1)
                    X = X.join(dums)
                
                if target_column in X.columns:
                    X = X.drop(target_column, axis=1)

                if target_column not in subset_df.columns:
                    self.logger.error(f"Target column {target_column} not found in dataframe")
                    continue

                y = subset_df[target_column].astype(float)
                
                if split_date is not None:
                    if isinstance(subset_df.index, pd.DatetimeIndex):
                        train_mask = subset_df.index <= split_date
                        test_mask = subset_df.index > split_date
                        X_train, X_test = X[train_mask], X[test_mask]
                        y_train, y_test = y[train_mask], y[test_mask]
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=params['test_size'], random_state=params['random_state']
                        )
                else:
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
                
                self.logger.info(f'Mean Absolute Error: {round(mae, 2)}')
                self.logger.info(f'Accuracy: {round(accuracy, 2)} %.')
                
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

        except Exception as e:
            self.logger.error(f"XGBoost train failed: {e}\n{traceback.format_exc()}")
            raise
            
        return output_dict

    def predict(self, 
                df: pd.DataFrame, 
                feature_columns: List[str],
                target_column: str,
                model_objects: Any, 
                context_date: Any,
                dataset_names: List[str],
                submode: str) -> Dict[str, Any]:
        
        output_dict = {'predict': {}}
        
        try:
            # Ensure model_objects is consistent (GenericForecast passes self.model)
            # In original code it was expecting 'model'
            model = model_objects

            # Resolve feature names from the model when feature_columns is empty
            # OR override feature_columns with the model's own features to avoid mismatch
            model_feature_names = None
            if hasattr(model, 'feature_names_in_'):
                model_feature_names = list(model.feature_names_in_)
            elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
                model_feature_names = model.get_booster().feature_names or []

            if model_feature_names:
                self.logger.debug(f"Using model feature names: {model_feature_names}")
                feature_columns = model_feature_names
            elif not feature_columns:
                self.logger.warning("No feature_columns and no model feature names")

            # Determine if model expects one-hot weekday columns
            _weekday_ohe_cols = [f'plug_in_weekday_{i}' for i in range(7)]
            _model_expects_ohe = model_feature_names and any(c in model_feature_names for c in _weekday_ohe_cols)
            # Raw columns to select from df (before one-hot expansion)
            _raw_cols = []
            if feature_columns:
                for c in feature_columns:
                    if c.startswith('plug_in_weekday_') and c not in df.columns:
                        # This is a one-hot column from the model; we need the raw plug_in_weekday
                        if 'plug_in_weekday' not in _raw_cols and 'plug_in_weekday' in df.columns:
                            _raw_cols.append('plug_in_weekday')
                    elif c in df.columns:
                        _raw_cols.append(c)
                # Deduplicate while preserving order
                _raw_cols = list(dict.fromkeys(_raw_cols))

            def _prepare_features(subset_df):
                """Select raw columns, one-hot encode weekday, reorder to model features."""
                if _raw_cols:
                    feats = subset_df[_raw_cols].copy()
                else:
                    feats = subset_df.select_dtypes(include='number').copy()
                # One-hot encode plug_in_weekday if the model expects it
                if 'plug_in_weekday' in feats.columns and _model_expects_ohe:
                    weekday_series = feats['plug_in_weekday']
                    dums = pd.get_dummies(weekday_series, prefix='plug_in_weekday')
                    dums = dums.reindex(columns=_weekday_ohe_cols, fill_value=0)
                    feats = feats.drop('plug_in_weekday', axis=1).join(dums)
                # Final column selection to match model exactly
                if model_feature_names:
                    missing = [c for c in model_feature_names if c not in feats.columns]
                    if missing:
                        self.logger.warning(f"[XGBoost.predict] Columns missing after prep: {missing}")
                    feats = feats.reindex(columns=model_feature_names, fill_value=0)
                return feats

            for dataset_name in dataset_names:
                if submode == 'schedule':
                    subset = df.loc[df['dataset_name'] == dataset_name]
                    if subset.empty:
                        continue
                    
                    input_row = subset.iloc[[-1]].copy()
                    features = _prepare_features(input_row)

                    prediction = model.predict(features)
                    val = float(prediction[0])
                    
                    output_dict['predict'].update({
                        self.output_key: val,
                        'value': val,
                        'date': datetime.now(),
                        'created_at': datetime.now()
                    })
                else:
                    subset = df.loc[df['dataset_name'] == dataset_name]
                    if subset.empty:
                        continue

                    # Extract dates
                    if isinstance(subset.index, pd.DatetimeIndex):
                        subset_dates = subset.index
                    elif 'plug_in_datetime' in subset.columns:
                        subset_dates = pd.DatetimeIndex(subset['plug_in_datetime'])
                    else:
                        subset_dates = None

                    features = _prepare_features(subset)

                    prediction = model.predict(features)
                    dates_list = subset_dates.strftime('%Y-%m-%d').tolist() if subset_dates is not None else None
                    actual_vals = subset[target_column].tolist() if target_column and target_column in subset.columns else None
                    output_dict['predict'].update({
                        self.output_key: prediction.tolist(),
                        'values': prediction.tolist(),
                        'dates': dates_list,
                        'actuals': actual_vals,
                        'date': context_date if context_date else datetime.now(),
                        'created_at': datetime.now()
                    })
        except Exception as e:
            self.logger.error(f"XGBoost predict failed: {e}\n{traceback.format_exc()}")
            raise

        return output_dict


class LSTMModelStrategy(ModelStrategy):
    def __init__(self, output_key: str = 'prediction'):
        self.output_key = output_key
        self.logger = logging.getLogger(__name__)

    def build_model(self, input_shape, learning_rate=0.001, activation='relu', dropout_rate=0.2):
        """
        Builds the LSTM model using the defined hyperparameters.
        """
        model = Sequential()
        
        model.add(Input(shape=input_shape))
        
        # First LSTM layer with Dropout
        model.add(KerasLSTM(units=64, activation=activation, return_sequences=True))
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer with Dropout
        model.add(KerasLSTM(units=32, activation=activation))
        model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Optimizer
        optimizer = Adam(learning_rate=learning_rate)
        
        # Error metrics: MAE, MSE
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        return model

    def train(self,
              df: pd.DataFrame,
              feature_columns: List[str],
              target_column: str,
              dataset_names: List[str],
              model_name_prefix: str,
              split_date: str = None) -> Dict[str, Any]:
        
        output_dict = {'train': {}}

        try:
            # For LSTM, we look for custom parameters or use defaults
            # We can extract these from dataframe attrs if passed, or fallback
            lstm_params = getattr(df, 'attrs', {}).get('lstm_params', {})
            epochs = lstm_params.get('epochs', 100)
            batch_size = lstm_params.get('batch_size', 32)
            learning_rate = lstm_params.get('learning_rate', 0.001)
            activation = lstm_params.get('activation', 'relu')
            dropout_rate = lstm_params.get('dropout_rate', 0.2)
            look_back = lstm_params.get('look_back', 30)

            for dataset_name in dataset_names:
                self.logger.info(f"{dataset_name} - LSTM Forecast ({target_column}) - Model training")
                
                subset_df = df.loc[df['dataset_name'] == dataset_name].copy()

                if target_column not in subset_df.columns:
                    self.logger.error(f"Target column {target_column} not found in dataframe")
                    continue

                if isinstance(subset_df.index, pd.DatetimeIndex):
                    train_mask = pd.Series(True, index=subset_df.index)
                    # Support explicit range matching tfm_forecast config
                    train_range = lstm_params.get('train_range')
                    if train_range:
                        start_date, end_date = train_range
                        if start_date:
                            train_mask &= (subset_df.index >= pd.to_datetime(start_date))
                        if end_date:
                            train_mask &= (subset_df.index <= pd.to_datetime(end_date))
                    else:
                        # Fallback behavior
                        train_start_date = lstm_params.get('train_start_date')
                        if train_start_date:
                            train_mask &= (subset_df.index >= pd.to_datetime(train_start_date))
                        
                        train_end_boundary = split_date or lstm_params.get('train_split_date')
                        if train_end_boundary:
                            train_mask &= (subset_df.index <= pd.to_datetime(train_end_boundary))
                        
                    train_data = subset_df[train_mask][[target_column]].values
                    data = train_data
                else:
                    data = subset_df[[target_column]].values
                
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)
                
                X, y = [], []
                for i in range(len(scaled_data) - look_back):
                    X.append(scaled_data[i:(i + look_back), 0])
                    y.append(scaled_data[i + look_back, 0])
                    
                X = np.array(X)
                y = np.array(y)
                
                if len(X) == 0:
                    self.logger.warning("Not enough data to train LSTM with current look_back.")
                    continue

                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                input_shape = (X.shape[1], 1)
                model = self.build_model(input_shape, learning_rate, activation, dropout_rate)
                
                self.logger.info(f"Training LSTM model for {epochs} epochs, batch size {batch_size}...")
                model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
                
                pilot_name = f"{model_name_prefix}_{dataset_name}"
                
                output_dict['train'][pilot_name] = {
                    'params': lstm_params,
                    'model': {
                        'keras_model': model,
                        'scaler': scaler,
                        'look_back': look_back
                    },
                    'metrics': {},
                    'artifacts': {}
                }

        except Exception as e:
            self.logger.error(f"LSTM train failed: {e}\n{traceback.format_exc()}")
            raise
            
        return output_dict

    def predict(self, 
                df: pd.DataFrame, 
                feature_columns: List[str],
                target_column: str,
                model_objects: Any, 
                context_date: Any,
                dataset_names: List[str],
                submode: str) -> Dict[str, Any]:
        
        output_dict = {'predict': {}}
        
        try:
            # We expect model_objects to be our dict with keras_model, scaler, etc.
            # But generic_forecast sometimes just passes the model or model dict
            if isinstance(model_objects, dict) and 'keras_model' in model_objects:
                model = model_objects['keras_model']
                scaler = model_objects['scaler']
                look_back = model_objects.get('look_back', 30)
                lstm_params = model_objects.get('params', {})
            else:
                self.logger.error("LSTMModelStrategy requires a dict with keras_model and scaler.")
                return output_dict

            prediction_days = lstm_params.get('prediction_days', 30)
            
            # Prediction boundaries if defined in params
            predict_start_date = lstm_params.get('predict_start_date')
            predict_end_date = lstm_params.get('predict_end_date')
            test_range = lstm_params.get('test_range', None)
            
            if test_range:
                predict_start_date = test_range[0] or predict_start_date
                predict_end_date = test_range[1] or predict_end_date

            for dataset_name in dataset_names:
                if submode == 'schedule':
                    subset = df.loc[df['dataset_name'] == dataset_name].copy()
                    if subset.empty:
                        continue
                    
                    # Apply specific boundaries if provided, otherwise respect the df explicitly given
                    if predict_start_date or predict_end_date:
                        if isinstance(subset.index, pd.DatetimeIndex):
                            mask = pd.Series(True, index=subset.index)
                            if predict_start_date:
                                mask = mask & (subset.index >= pd.to_datetime(predict_start_date))
                            if predict_end_date:
                                mask = mask & (subset.index <= pd.to_datetime(predict_end_date))
                            subset = subset[mask]
                            
                    if len(subset) < look_back:
                        self.logger.warning(f"Not enough data for {dataset_name} to fulfill look_back of {look_back}")
                        continue

                    data = subset[[target_column]].values[-look_back:]
                    scaled_data = scaler.transform(data)
                    
                    current_seq = scaled_data.copy()
                    predictions = []
                    
                    for _ in range(prediction_days):
                        pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)
                        predictions.append(pred[0, 0])
                        
                        current_seq = np.roll(current_seq, -1, axis=0)
                        current_seq[-1, 0] = pred[0, 0]
                    
                    predictions_unscaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                    predictions_series = predictions_unscaled.flatten().tolist()
                    
                    # Generate future dates from last data point
                    if isinstance(subset.index, pd.DatetimeIndex):
                        last_date = subset.index[-1]
                    elif 'plug_in_datetime' in subset.columns:
                        last_date = pd.to_datetime(subset['plug_in_datetime']).iloc[-1]
                    else:
                        last_date = pd.Timestamp.now()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                                 periods=prediction_days, freq='D')

                    output_dict['predict'].update({
                        self.output_key: predictions_series,
                        'values': predictions_series,
                        'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                        'value': predictions_series[0] if predictions_series else 0,
                        'date': datetime.now(),
                        'created_at': datetime.now()
                    })

        except Exception as e:
            self.logger.error(f"LSTM predict failed: {e}\n{traceback.format_exc()}")
            raise

        return output_dict


class TransformerModelStrategy(ModelStrategy):
    def __init__(self, output_key: str = 'prediction'):
        self.output_key = output_key
        self.logger = logging.getLogger(__name__)

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, 
                               num_heads=num_heads, 
                               dropout=dropout
                               )(x, x)
        x = Dropout(dropout)(x)
        res = Add()([x, inputs])

        # Feed Forward Part
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        return Add()([x, res])

    def build_model(
        self,
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        dropout=0.1,
        mlp_dropout=0.1,
        learning_rate=0.001
    ):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(1)(x)

        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])
        return model

    def train(self,
              df: pd.DataFrame,
              feature_columns: List[str],
              target_column: str,
              dataset_names: List[str],
              model_name_prefix: str,
              split_date: str = None) -> Dict[str, Any]:
        output_dict = {'train': {}}

        try:
            transformer_params = getattr(df, 'attrs', {}).get('transformer_params', {})
            epochs = transformer_params.get('epochs', 100)
            batch_size = transformer_params.get('batch_size', 32)
            learning_rate = transformer_params.get('learning_rate', 0.001)
            look_back = transformer_params.get('look_back', 30)

            head_size = transformer_params.get('head_size', 256)
            num_heads = transformer_params.get('num_heads', 4)
            ff_dim = transformer_params.get('ff_dim', 4)
            num_transformer_blocks = transformer_params.get('num_transformer_blocks', 4)
            mlp_units = transformer_params.get('mlp_units', [128])
            dropout = transformer_params.get('dropout', 0.1)
            mlp_dropout = transformer_params.get('mlp_dropout', 0.1)

            for dataset_name in dataset_names:
                self.logger.info(f"{dataset_name} - Transformer Forecast ({target_column}) - Model training")
                subset_df = df.loc[df['dataset_name'] == dataset_name].copy()

                if target_column not in subset_df.columns:
                    self.logger.error(f"Target column {target_column} not found in dataframe")
                    continue

                if isinstance(subset_df.index, pd.DatetimeIndex):
                    train_mask = pd.Series(True, index=subset_df.index)
                    train_range = transformer_params.get('train_range')
                    if train_range:
                        start_date, end_date = train_range
                        if start_date:
                            train_mask &= (subset_df.index >= pd.to_datetime(start_date))
                        if end_date:
                            train_mask &= (subset_df.index <= pd.to_datetime(end_date))
                    else:
                        train_start_date = transformer_params.get('train_start_date')
                        if train_start_date:
                            train_mask &= (subset_df.index >= pd.to_datetime(train_start_date))
                        train_end_boundary = split_date or transformer_params.get('train_split_date')
                        if train_end_boundary:
                            train_mask &= (subset_df.index <= pd.to_datetime(train_end_boundary))
                        
                    train_data = subset_df[train_mask][[target_column]].values
                    data = train_data
                else:
                    data = subset_df[[target_column]].values
                
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)
                
                X, y = [], []
                for i in range(len(scaled_data) - look_back):
                    X.append(scaled_data[i:(i + look_back), 0])
                    y.append(scaled_data[i + look_back, 0])
                    
                X = np.array(X)
                y = np.array(y)
                
                if len(X) == 0:
                    self.logger.warning("Not enough data to train Transformer with current look_back.")
                    continue

                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                input_shape = (X.shape[1], 1)
                
                model = self.build_model(
                    input_shape=input_shape,
                    head_size=head_size,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    num_transformer_blocks=num_transformer_blocks,
                    mlp_units=mlp_units,
                    dropout=dropout,
                    mlp_dropout=mlp_dropout,
                    learning_rate=learning_rate
                )
                
                self.logger.info(f"Training Transformer model for {epochs} epochs, batch size {batch_size}...")
                model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
                
                pilot_name = f"{model_name_prefix}_{dataset_name}"
                
                output_dict['train'][pilot_name] = {
                    'params': transformer_params,
                    'model': {
                        'keras_model': model,
                        'scaler': scaler,
                        'look_back': look_back
                    },
                    'metrics': {},
                    'artifacts': {}
                }

        except Exception as e:
            self.logger.error(f"Transformer train failed: {e}\n{traceback.format_exc()}")
            raise
            
        return output_dict

    def predict(self, 
                df: pd.DataFrame, 
                feature_columns: List[str],
                target_column: str,
                model_objects: Any, 
                context_date: Any,
                dataset_names: List[str],
                submode: str) -> Dict[str, Any]:
        
        output_dict = {'predict': {}}
        
        try:
            if isinstance(model_objects, dict) and 'keras_model' in model_objects:
                model = model_objects['keras_model']
                scaler = model_objects['scaler']
                look_back = model_objects.get('look_back', 30)
                transformer_params = model_objects.get('params', {})
            else:
                self.logger.error("TransformerModelStrategy requires a dict with keras_model and scaler.")
                return output_dict

            prediction_days = transformer_params.get('prediction_days', 30)
            predict_start_date = transformer_params.get('predict_start_date')
            predict_end_date = transformer_params.get('predict_end_date')
            test_range = transformer_params.get('test_range', None)
            
            if test_range:
                predict_start_date = test_range[0] or predict_start_date
                predict_end_date = test_range[1] or predict_end_date

            for dataset_name in dataset_names:
                if submode == 'schedule':
                    subset = df.loc[df['dataset_name'] == dataset_name].copy()
                    if subset.empty:
                        continue
                    
                    if predict_start_date or predict_end_date:
                        if isinstance(subset.index, pd.DatetimeIndex):
                            mask = pd.Series(True, index=subset.index)
                            if predict_start_date:
                                mask = mask & (subset.index >= pd.to_datetime(predict_start_date))
                            if predict_end_date:
                                mask = mask & (subset.index <= pd.to_datetime(predict_end_date))
                            subset = subset[mask]
                            
                    if len(subset) < look_back:
                        self.logger.warning(f"Not enough data for {dataset_name} to fulfill look_back of {look_back}")
                        continue

                    data = subset[[target_column]].values[-look_back:]
                    scaled_data = scaler.transform(data)
                    
                    current_seq = scaled_data.copy()
                    predictions = []
                    
                    for _ in range(prediction_days):
                        pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)
                        predictions.append(pred[0, 0])
                        current_seq = np.roll(current_seq, -1, axis=0)
                        current_seq[-1, 0] = pred[0, 0]
                    
                    predictions_unscaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                    predictions_series = predictions_unscaled.flatten().tolist()
                    
                    if isinstance(subset.index, pd.DatetimeIndex):
                        last_date = subset.index[-1]
                    elif 'plug_in_datetime' in subset.columns:
                        last_date = pd.to_datetime(subset['plug_in_datetime']).iloc[-1]
                    else:
                        last_date = pd.Timestamp.now()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')

                    output_dict['predict'].update({
                        self.output_key: predictions_series,
                        'values': predictions_series,
                        'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                        'value': predictions_series[0] if predictions_series else 0,
                        'date': datetime.now(),
                        'created_at': datetime.now()
                    })

        except Exception as e:
            self.logger.error(f"Transformer predict failed: {e}\n{traceback.format_exc()}")
            raise

        return output_dict
class HybridTransformerLSTMModelStrategy(ModelStrategy):
    def __init__(self, output_key: str = 'prediction'):
        self.output_key = output_key
        self.logger = logging.getLogger(__name__)

    def build_model(
        self,
        input_shape,
        d_model=128,
        num_heads=4,
        dropout=0.1,
        learning_rate=0.001
    ):
        inputs = Input(shape=input_shape)
        
        # Encoder: Single LSTM layer processing the sequence
        lstm_enc = KerasLSTM(units=d_model, return_sequences=True)(inputs)
        
        # Multi-Head Attention, followed by Layer Normalization and Dropout
        attn_enc = MultiHeadAttention(key_dim=d_model, num_heads=num_heads, dropout=dropout)(lstm_enc, lstm_enc)
        enc_add = Add()([attn_enc, lstm_enc])
        enc_out = LayerNormalization(epsilon=1e-6)(enc_add)
        
        # Decoder: Extra LSTM layer extracting from the encoder output sequence context
        lstm_dec = KerasLSTM(units=d_model, return_sequences=True)(enc_out)
        
        # Multi-Head Attention
        attn_dec = MultiHeadAttention(key_dim=d_model, num_heads=num_heads, dropout=dropout)(lstm_dec, lstm_dec)
        dec_add = Add()([attn_dec, lstm_dec])
        dec_out = LayerNormalization(epsilon=1e-6)(dec_add)
        
        # A flattened layer is used to predict the output sequence
        flat = Flatten()(dec_out)
        outputs = Dense(1)(flat)
        
        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])
        return model

    def train(self,
              df: pd.DataFrame,
              feature_columns: List[str],
              target_column: str,
              dataset_names: List[str],
              model_name_prefix: str,
              split_date: str = None) -> Dict[str, Any]:
        output_dict = {'train': {}}

        try:
            hybrid_params = getattr(df, 'attrs', {}).get('hybrid_params', {})
            epochs = hybrid_params.get('epochs', 100)
            batch_size = hybrid_params.get('batch_size', 32)
            learning_rate = hybrid_params.get('learning_rate', 0.001)
            look_back = hybrid_params.get('look_back', 30)

            d_model = hybrid_params.get('d_model', 128)
            num_heads = hybrid_params.get('num_heads', 4)
            dropout = hybrid_params.get('dropout', 0.1)

            for dataset_name in dataset_names:
                self.logger.info(f"{dataset_name} - Hybrid Transformer-LSTM Forecast ({target_column}) - Model training")
                subset_df = df.loc[df['dataset_name'] == dataset_name].copy()

                if target_column not in subset_df.columns:
                    self.logger.error(f"Target column {target_column} not found in dataframe")
                    continue

                if isinstance(subset_df.index, pd.DatetimeIndex):
                    train_mask = pd.Series(True, index=subset_df.index)
                    train_range = hybrid_params.get('train_range')
                    if train_range:
                        start_date, end_date = train_range
                        if start_date:
                            train_mask &= (subset_df.index >= pd.to_datetime(start_date))
                        if end_date:
                            train_mask &= (subset_df.index <= pd.to_datetime(end_date))
                    else:
                        train_start_date = hybrid_params.get('train_start_date')
                        if train_start_date:
                            train_mask &= (subset_df.index >= pd.to_datetime(train_start_date))
                        train_end_boundary = split_date or hybrid_params.get('train_split_date')
                        if train_end_boundary:
                            train_mask &= (subset_df.index <= pd.to_datetime(train_end_boundary))
                        
                    train_data = subset_df[train_mask][[target_column]].values
                    data = train_data
                else:
                    data = subset_df[[target_column]].values
                
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)
                
                X, y = [], []
                for i in range(len(scaled_data) - look_back):
                    X.append(scaled_data[i:(i + look_back), 0])
                    y.append(scaled_data[i + look_back, 0])
                    
                X = np.array(X)
                y = np.array(y)
                
                if len(X) == 0:
                    self.logger.warning("Not enough data to train Hybrid Transformer-LSTM with current look_back.")
                    continue

                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                input_shape = (X.shape[1], X.shape[2])
                
                model = self.build_model(
                    input_shape=input_shape,
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    learning_rate=learning_rate
                )
                
                self.logger.info(f"Training Hybrid Transformer-LSTM model for {epochs} epochs, batch size {batch_size}...")
                model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
                
                pilot_name = f"{model_name_prefix}_{dataset_name}"
                
                output_dict['train'][pilot_name] = {
                    'params': hybrid_params,
                    'model': {
                        'keras_model': model,
                        'scaler': scaler,
                        'look_back': look_back
                    },
                    'metrics': {},
                    'artifacts': {}
                }

        except Exception as e:
            self.logger.error(f"Hybrid Transformer-LSTM train failed: {e}\n{traceback.format_exc()}")
            raise
            
        return output_dict

    def predict(self, 
                df: pd.DataFrame, 
                feature_columns: List[str],
                target_column: str,
                model_objects: Any, 
                context_date: Any,
                dataset_names: List[str],
                submode: str) -> Dict[str, Any]:
        
        output_dict = {'predict': {}}
        
        try:
            if isinstance(model_objects, dict) and 'keras_model' in model_objects:
                model = model_objects['keras_model']
                scaler = model_objects['scaler']
                look_back = model_objects.get('look_back', 30)
                hybrid_params = model_objects.get('params', {})
            else:
                self.logger.error("HybridTransformerLSTMModelStrategy requires a dict with keras_model and scaler.")
                return output_dict

            prediction_days = hybrid_params.get('prediction_days', 30)
            predict_start_date = hybrid_params.get('predict_start_date')
            predict_end_date = hybrid_params.get('predict_end_date')
            test_range = hybrid_params.get('test_range', None)
            
            if test_range:
                predict_start_date = test_range[0] or predict_start_date
                predict_end_date = test_range[1] or predict_end_date

            for dataset_name in dataset_names:
                if submode == 'schedule':
                    subset = df.loc[df['dataset_name'] == dataset_name].copy()
                    if subset.empty:
                        continue
                    
                    if predict_start_date or predict_end_date:
                        if isinstance(subset.index, pd.DatetimeIndex):
                            mask = pd.Series(True, index=subset.index)
                            if predict_start_date:
                                mask = mask & (subset.index >= pd.to_datetime(predict_start_date))
                            if predict_end_date:
                                mask = mask & (subset.index <= pd.to_datetime(predict_end_date))
                            subset = subset[mask]
                            
                    if len(subset) < look_back:
                        self.logger.warning(f"Not enough data for {dataset_name} to fulfill look_back of {look_back}")
                        continue

                    data = subset[[target_column]].values[-look_back:]
                    scaled_data = scaler.transform(data)
                    
                    current_seq = scaled_data.copy()
                    predictions = []
                    
                    for _ in range(prediction_days):
                        pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)
                        predictions.append(pred[0, 0])
                        current_seq = np.roll(current_seq, -1, axis=0)
                        current_seq[-1, 0] = pred[0, 0]
                    
                    predictions_unscaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                    predictions_series = predictions_unscaled.flatten().tolist()
                    
                    if isinstance(subset.index, pd.DatetimeIndex):
                        last_date = subset.index[-1]
                    elif 'plug_in_datetime' in subset.columns:
                        last_date = pd.to_datetime(subset['plug_in_datetime']).iloc[-1]
                    else:
                        last_date = pd.Timestamp.now()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')

                    output_dict['predict'].update({
                        self.output_key: predictions_series,
                        'values': predictions_series,
                        'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                        'value': predictions_series[0] if predictions_series else 0,
                        'date': datetime.now(),
                        'created_at': datetime.now()
                    })

        except Exception as e:
            self.logger.error(f"Hybrid Transformer-LSTM predict failed: {e}\n{traceback.format_exc()}")
            raise

        return output_dict
