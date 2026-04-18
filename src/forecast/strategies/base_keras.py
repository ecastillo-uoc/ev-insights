"""
Base strategy for Keras-based time-series forecasting models (LSTM, Transformer, Hybrid).

Extracts the shared train/predict pipeline that is identical across all three neural
model strategies:

  * Date-range filtering and train/test boundary resolution
  * MinMaxScaler normalisation + sliding-window sequence construction
  * Schedule-mode multi-step recursive prediction
  * Backtest-mode rolling one-step prediction
  * Output-dict construction (params, model bundle, metrics / predictions)

Subclasses only need to implement:
  - ``build_model(input_shape, **kwargs)`` — architecture-specific Keras model
  - ``_params_key`` property — the ``df.attrs`` key for strategy-specific hyperparams
  - ``_strategy_display_name`` property — human-readable name used in log messages
"""

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import logging
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from .interfaces import ModelStrategy


class KerasTimeSeriesBaseStrategy(ModelStrategy):
    """Template-method base for univariate Keras time-series strategies.

    The ``train()`` and ``predict()`` methods implement the full pipeline using
    helper methods that encapsulate each duplicated block.  Subclasses override
    only ``build_model`` and the two name-related properties.
    """

    def __init__(self, output_key: str = 'prediction'):
        self.output_key = output_key
        self.logger = logging.getLogger(self.__class__.__module__)

    # ------------------------------------------------------------------
    # Abstract members — subclasses MUST implement
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def _params_key(self) -> str:
        """Key used to retrieve hyperparams from ``df.attrs``, e.g. ``'lstm_params'``."""

    @property
    @abstractmethod
    def _strategy_display_name(self) -> str:
        """Human-readable strategy name for log messages, e.g. ``'LSTM'``."""

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], **kwargs):
        """Construct and compile a Keras ``Model`` for the given *input_shape*."""

    # ------------------------------------------------------------------
    # Shared train helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_train_data(
        subset_df: pd.DataFrame,
        target_column: str,
        params: dict,
        split_date: str | None,
    ) -> np.ndarray:
        """Return the 2-D numpy array of training target values.

        Applies ``train_range`` or individual start/end boundaries, falling
        back to the full column when no datetime index is available.
        """
        if isinstance(subset_df.index, pd.DatetimeIndex):
            train_mask = pd.Series(True, index=subset_df.index)
            train_range = params.get('train_range')
            if train_range:
                start_date, end_date = train_range
                if start_date:
                    train_mask &= (subset_df.index >= pd.to_datetime(start_date))
                if end_date:
                    train_mask &= (subset_df.index <= pd.to_datetime(end_date))
            else:
                train_start_date = params.get('train_start_date')
                if train_start_date:
                    train_mask &= (subset_df.index >= pd.to_datetime(train_start_date))
                train_end_boundary = split_date or params.get('train_split_date')
                if train_end_boundary:
                    train_mask &= (subset_df.index <= pd.to_datetime(train_end_boundary))
            return subset_df[train_mask][[target_column]].values
        return subset_df[[target_column]].values

    @staticmethod
    def _scale_and_create_sequences(
        data: np.ndarray,
        look_back: int,
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """MinMax-scale *data* and build sliding-window sequences.

        Returns ``(X, y, scaler)`` where *X* has shape
        ``(n_samples, look_back, 1)`` ready for Keras models.
        """
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:(i + look_back), 0])
            y.append(scaled_data[i + look_back, 0])
        X = np.array(X)
        y = np.array(y)

        if len(X) > 0:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, scaler

    # ------------------------------------------------------------------
    # Shared predict helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_model_objects(
        model_objects: Any,
        strategy_name: str,
        logger: logging.Logger,
    ) -> Tuple[Any, MinMaxScaler, int, int | None, dict] | None:
        """Extract ``(model, scaler, look_back, forecast_horizon, params)`` from *model_objects*.

        Returns ``None`` when the expected dict layout is not found.
        """
        if isinstance(model_objects, dict) and 'keras_model' in model_objects:
            return (
                model_objects['keras_model'],
                model_objects['scaler'],
                model_objects.get('look_back', 30),
                model_objects.get('forecast_horizon'),
                model_objects.get('params', {}),
            )
        logger.error(f"{strategy_name} requires a dict with keras_model and scaler.")
        return None

    @staticmethod
    def _resolve_predict_bounds(params: dict, forecast_horizon: int | None = None) -> Tuple[int, str | None, str | None]:
        """Return ``(forecast_horizon, predict_start_date, predict_end_date)``.


        """
        default_days = forecast_horizon if forecast_horizon is not None else 30
        forecast_horizon = params.get('forecast_horizon', default_days)
        predict_start_date = params.get('predict_start_date')
        predict_end_date = params.get('predict_end_date')
        test_range = params.get('test_range')
        if test_range:
            predict_start_date = test_range[0] or predict_start_date
            predict_end_date = test_range[1] or predict_end_date
        return forecast_horizon, predict_start_date, predict_end_date

    def _predict_schedule(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_names: List[str],
        model,
        scaler: MinMaxScaler,
        look_back: int,
        forecast_horizon: int,
        predict_start_date: str | None,
        predict_end_date: str | None,
    ) -> dict:
        """Run **schedule** (multi-step recursive) prediction for each dataset."""
        output_dict: Dict[str, Any] = {}

        for dataset_name in dataset_names:
            subset = df.loc[df['dataset_name'] == dataset_name].copy()
            if subset.empty:
                continue

            # Apply date boundaries
            if predict_start_date or predict_end_date:
                if isinstance(subset.index, pd.DatetimeIndex):
                    mask = pd.Series(True, index=subset.index)
                    if predict_start_date:
                        mask = mask & (subset.index >= pd.to_datetime(predict_start_date))
                    if predict_end_date:
                        mask = mask & (subset.index <= pd.to_datetime(predict_end_date))
                    subset = subset[mask]

            if len(subset) < look_back:
                self.logger.warning(
                    f"Not enough data for {dataset_name} to fulfill look_back of {look_back}"
                )
                continue

            data = subset[[target_column]].values[-look_back:]
            scaled_data = scaler.transform(data)
            current_seq = scaled_data.copy()
            predictions = []

            for _ in range(forecast_horizon):
                pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)
                predictions.append(pred[0, 0])
                current_seq = np.roll(current_seq, -1, axis=0)
                current_seq[-1, 0] = pred[0, 0]

            predictions_unscaled = scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            )
            predictions_series = predictions_unscaled.flatten().tolist()

            # Determine last date for future date range
            if isinstance(subset.index, pd.DatetimeIndex):
                last_date = subset.index[-1]
            elif 'plug_in_datetime' in subset.columns:
                last_date = pd.to_datetime(subset['plug_in_datetime']).iloc[-1]
            else:
                last_date = pd.Timestamp.now()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq='D',
            )

            output_dict.update({
                self.output_key: predictions_series,
                'values': predictions_series,
                'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'value': predictions_series[0] if predictions_series else 0,
                'date': datetime.now(),
                'created_at': datetime.now(),
            })

        return output_dict

    def _predict_backtest(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_names: List[str],
        model,
        scaler: MinMaxScaler,
        look_back: int,
    ) -> dict:
        """Run **backtest** (rolling one-step) prediction for each dataset."""
        output_dict: Dict[str, Any] = {}

        for dataset_name in dataset_names:
            subset = df.loc[df['dataset_name'] == dataset_name].copy()
            if subset.empty:
                continue

            all_data = subset[[target_column]].values
            scaled_all = scaler.transform(all_data)

            if len(scaled_all) <= look_back:
                self.logger.warning(
                    f"Not enough data for {dataset_name} to backtest with look_back={look_back}"
                )
                continue

            predictions_scaled = []
            for i in range(look_back, len(scaled_all)):
                seq = scaled_all[i - look_back:i].reshape(1, look_back, 1)
                pred = model.predict(seq, verbose=0)
                predictions_scaled.append(pred[0, 0])

            predictions_unscaled = scaler.inverse_transform(
                np.array(predictions_scaled).reshape(-1, 1)
            ).flatten().tolist()

            actuals_unscaled = all_data[look_back:].flatten().tolist()

            if isinstance(subset.index, pd.DatetimeIndex):
                pred_dates = subset.index[look_back:].strftime('%Y-%m-%d').tolist()
            else:
                pred_dates = list(range(len(predictions_unscaled)))

            output_dict.update({
                self.output_key: predictions_unscaled,
                'values': predictions_unscaled,
                'actuals': actuals_unscaled,
                'dates': pred_dates,
                'value': predictions_unscaled[0] if predictions_unscaled else 0,
                'date': datetime.now(),
                'created_at': datetime.now(),
            })

        return output_dict

    # ------------------------------------------------------------------
    # Template-method: train
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        dataset_names: List[str],
        model_name_prefix: str,
        split_date: str = None,
    ) -> Dict[str, Any]:
        output_dict: Dict[str, Any] = {'train': {}}

        try:
            strategy_params = getattr(df, 'attrs', {}).get(self._params_key, {})
            epochs = strategy_params.get('epochs', 100)
            batch_size = strategy_params.get('batch_size', 32)
            look_back = strategy_params.get('look_back', 30)
            forecast_horizon = strategy_params.get('forecast_horizon')

            for dataset_name in dataset_names:
                self.logger.info(
                    f"{dataset_name} - {self._strategy_display_name} Forecast "
                    f"({target_column}) - Model training"
                )
                subset_df = df.loc[df['dataset_name'] == dataset_name].copy()

                if target_column not in subset_df.columns:
                    self.logger.error(
                        f"Target column {target_column} not found in dataframe"
                    )
                    continue

                # --- Filter + scale + window ---
                data = self._filter_train_data(
                    subset_df, target_column, strategy_params, split_date,
                )
                X, y, scaler = self._scale_and_create_sequences(data, look_back)

                if len(X) == 0:
                    self.logger.warning(
                        f"Not enough data to train {self._strategy_display_name} "
                        f"with current look_back."
                    )
                    continue

                # --- Build & fit model (architecture-specific) ---
                input_shape = (X.shape[1], X.shape[2])
                model = self._build_and_fit(
                    input_shape, X, y, epochs, batch_size, strategy_params,
                )

                # --- Store output ---
                pilot_name = f"{model_name_prefix}_{dataset_name}"
                output_dict['train'][pilot_name] = {
                    'params': strategy_params,
                    'model': {
                        'keras_model': model,
                        'scaler': scaler,
                        'look_back': look_back,
                        'forecast_horizon': forecast_horizon,
                    },
                    'metrics': {},
                    'artifacts': {},
                }

        except Exception as e:
            self.logger.error(
                f"{self._strategy_display_name} train failed: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

        return output_dict

    def _build_and_fit(
        self,
        input_shape: Tuple[int, ...],
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        strategy_params: dict,
    ):
        """Build the model via the subclass hook and run ``model.fit()``.

        Override in subclasses that need to pass extra architecture-specific
        keyword arguments to ``build_model()``.
        """
        model = self.build_model(input_shape, **strategy_params)

        # Optional Keras callbacks — enabled by strategy_params flags.
        # Hussain et al. (2025) use ReduceLROnPlateau and EarlyStopping.
        callbacks = []
        if strategy_params.get('use_lr_scheduler', False):
            callbacks.append(ReduceLROnPlateau(
                monitor='loss', patience=10, factor=0.5, min_lr=1e-6, verbose=0,
            ))
        if strategy_params.get('use_early_stopping', False):
            callbacks.append(EarlyStopping(
                monitor='loss', patience=20, restore_best_weights=True, verbose=0,
            ))

        self.logger.info(
            f"Training {self._strategy_display_name} model for {epochs} epochs, "
            f"batch size {batch_size}..."
        )
        model.fit(
            X, y, epochs=epochs, batch_size=batch_size, verbose=0,
            callbacks=callbacks if callbacks else None,
        )
        return model

    # ------------------------------------------------------------------
    # Template-method: predict
    # ------------------------------------------------------------------

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
        output_dict: Dict[str, Any] = {'predict': {}}

        try:
            unpacked = self._unpack_model_objects(
                model_objects, self._strategy_display_name, self.logger,
            )
            if unpacked is None:
                return output_dict
            model, scaler, look_back, forecast_horizon, params = unpacked

            forecast_horizon, predict_start_date, predict_end_date = (
                self._resolve_predict_bounds(params, forecast_horizon)
            )

            if submode == 'schedule':
                result = self._predict_schedule(
                    df, target_column, dataset_names,
                    model, scaler, look_back,
                    forecast_horizon, predict_start_date, predict_end_date,
                )
            else:
                result = self._predict_backtest(
                    df, target_column, dataset_names,
                    model, scaler, look_back,
                )

            output_dict['predict'].update(result)

        except Exception as e:
            self.logger.error(
                f"{self._strategy_display_name} predict failed: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

        return output_dict
