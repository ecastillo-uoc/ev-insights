"""
XGBoost gradient-boosted tree strategy for tabular EV charging forecasts.

Trains an ``XGBRegressor`` with ``reg:squarederror`` objective.  Handles
one-hot encoding of ``plug_in_weekday`` transparently and resolves feature
columns from the fitted model at predict time.

Hyperparameters ``random_state`` and ``test_size`` can be overridden via
``df.attrs['xgb_params']``.
"""

from typing import Any, Dict, List

import logging
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .interfaces import ModelStrategy


class XGBoostModelStrategy(ModelStrategy):
    """XGBoost regressor strategy with automatic weekday one-hot encoding."""

    def __init__(self, output_key: str = 'prediction'):
        self.output_key = output_key
        self.logger = logging.getLogger(__name__)

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        dataset_names: List[str],
        model_name_prefix: str,
        split_date: str = None,
    ) -> Dict[str, Any]:

        output_dict = {'train': {}}
        user_xgb_params = df.attrs.get('xgb_params', {})
        params = {
            'random_state': user_xgb_params.get('random_state', 16),
            'test_size': user_xgb_params.get('test_size', 0.20),
        }

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

    def predict(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        model_objects: Any,
        context_date: Any,
        dataset_names: List[str],
        submode: str,
    ) -> Dict[str, Any]:

        output_dict = {'predict': {}}

        try:
            model = model_objects

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

            _weekday_ohe_cols = [f'plug_in_weekday_{i}' for i in range(7)]
            _model_expects_ohe = model_feature_names and any(c in model_feature_names for c in _weekday_ohe_cols)
            _raw_cols: list = []
            if feature_columns:
                for c in feature_columns:
                    if c.startswith('plug_in_weekday_') and c not in df.columns:
                        if 'plug_in_weekday' not in _raw_cols and 'plug_in_weekday' in df.columns:
                            _raw_cols.append('plug_in_weekday')
                    elif c in df.columns:
                        _raw_cols.append(c)
                _raw_cols = list(dict.fromkeys(_raw_cols))

            def _prepare_features(subset_df):
                """Select raw columns, one-hot encode weekday, reorder to model features."""
                if _raw_cols:
                    feats = subset_df[_raw_cols].copy()
                else:
                    feats = subset_df.select_dtypes(include='number').copy()
                if 'plug_in_weekday' in feats.columns and _model_expects_ohe:
                    weekday_series = feats['plug_in_weekday']
                    dums = pd.get_dummies(weekday_series, prefix='plug_in_weekday')
                    dums = dums.reindex(columns=_weekday_ohe_cols, fill_value=0)
                    feats = feats.drop('plug_in_weekday', axis=1).join(dums)
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
