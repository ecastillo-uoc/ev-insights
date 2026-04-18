"""
LightGBM gradient-boosted tree strategy for tabular EV charging forecasts.

Uses the LightGBM ``lgb.train()`` API with configurable hyperparameters
(``num_leaves``, ``learning_rate``, ``max_depth``, ``early_stopping_rounds``)
read from ``df.attrs['lgbm_params']``.

Supports both date-based chronological and percentage-based train/test splits.
"""

from typing import Any, Dict, List

import logging
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

from .interfaces import ModelStrategy
from .utils_ts import smape


class LightGBMModelStrategy(ModelStrategy):
    """LightGBM regressor strategy with early stopping and SMAPE/MAPE metrics."""

    def __init__(self, output_key: str = 'prediction'):
        self.output_key = output_key
        self.logger = logging.getLogger(__name__)

    def train(self, df, feature_columns, target_column, dataset_names, model_name_prefix, split_date=None):
        from ..forecast import Forecast
        output_dict = {'train': {}}
        params = {'random_state': 16, 'test_size': 0.20}

        try:
            user_lgbm_params = df.attrs.get('lgbm_params', {})

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
                    if isinstance(subset_df.index, pd.DatetimeIndex):
                        train_mask = subset_df.index <= split_date
                        test_mask = subset_df.index > split_date
                    else:
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

                X_train = X_train.select_dtypes(exclude=['object', 'string'])
                X_test = X_test.select_dtypes(exclude=['object', 'string'])
                features_to_use = list(X_train.columns)

                lgb_params = {
                    'num_leaves': user_lgbm_params.get('num_leaves', 10),
                    'learning_rate': user_lgbm_params.get('learning_rate', 0.02),
                    'max_depth': user_lgbm_params.get('max_depth', 5),
                    'verbose': 0,
                    'early_stopping_rounds': user_lgbm_params.get('early_stopping_rounds', 200),
                    'nthread': -1
                }
                params = {**params, **{k: v for k, v in lgb_params.items() if k != 'verbose' and k != 'nthread'}}

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
            debug_msg = (
                f"[LightGBM.predict] Starting predict: "
                f"df.shape={df.shape if df is not None else None}, "
                f"feature_columns={feature_columns}, "
                f"target_column={target_column}, "
                f"model_objects type={type(model_objects).__name__}, "
                f"context_date={context_date}, "
                f"dataset_names={dataset_names}, "
                f"submode={submode}"
            )
            self.logger.debug(debug_msg)

            model = model_objects
            if model is None:
                self.logger.error("[LightGBM.predict] model_objects is None – cannot predict")
                return output_dict

            if hasattr(model, 'feature_name'):
                model_features = model.feature_name()
                self.logger.debug(f"[LightGBM.predict] Model trained features ({len(model_features)}): {model_features}")
                missing = [f for f in model_features if f not in df.columns]
                if missing:
                    self.logger.error(f"[LightGBM.predict] Missing features in df: {missing}")
                feature_columns = model_features

            for dataset_name in (dataset_names or []):
                self.logger.debug(f"[LightGBM.predict] Processing dataset: {dataset_name}")
                subset = df.loc[df['dataset_name'] == dataset_name]
                self.logger.debug(f"[LightGBM.predict] Subset rows: {len(subset)}")

                if subset.empty:
                    self.logger.warning(f"[LightGBM.predict] Empty subset for {dataset_name}, skipping")
                    continue

                if isinstance(subset.index, pd.DatetimeIndex):
                    subset_dates = subset.index
                elif 'plug_in_datetime' in subset.columns:
                    subset_dates = pd.DatetimeIndex(subset['plug_in_datetime'])
                else:
                    subset_dates = None

                if submode == 'schedule':
                    input_row = subset.iloc[[-1]].copy()
                    X = input_row[feature_columns]
                    self.logger.debug(
                        f"[LightGBM.predict] Schedule mode, input_row features: "
                        f"{X.columns.tolist()}, values: {X.values.tolist()}"
                    )
                else:
                    X = subset[feature_columns]
                    self.logger.debug(f"[LightGBM.predict] Full mode, X.shape={X.shape}")

                prediction = model.predict(
                    X, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None
                )
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
