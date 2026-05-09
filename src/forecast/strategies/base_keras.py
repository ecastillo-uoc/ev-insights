"""
Base strategy for Keras-based time-series forecasting models (LSTM, Transformer, Hybrid).

Extracts the shared train/predict pipeline that is identical across all three neural
model strategies:

  * Date-range filtering and train/test boundary resolution
  * RobustScaler normalisation + sliding-window sequence construction (with ±3.0 clip guard)
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
from sklearn.preprocessing import RobustScaler

# Ensure GPU is configured before any model is built (idempotent: no-op on
# repeated calls or when running without a GPU).
from src.utils.gpu_config import configure_gpu as _configure_gpu
_configure_gpu()

from .interfaces import ModelStrategy


def _sinusoidal_pe(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding matrix of shape ``(seq_len, d_model)``.

    Follows Vaswani et al. (2017) Eq. 3::

      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Parameterised by *seq_len* (= look_back) so the matrix always matches
    the context window used at training and prediction time.
    """
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10_000.0, (2 * (dims // 2)) / d_model)
    angle_rads = positions * angle_rates
    pe = np.zeros_like(angle_rads)
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pe.astype(np.float32)


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

    @property
    def _scaler_class(self):
        """Sklearn scaler class used to normalise sequences.

        Override in subclasses to use a different scaler.
        Default: ``RobustScaler`` (our models).
        Hussain et al. variants override this to ``MinMaxScaler``.
        """
        return RobustScaler

    # ------------------------------------------------------------------
    # Shared train helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_train_data(
        subset_df: pd.DataFrame,
        target_column: str,
        params: dict,
        split_date: str | None,
        feature_columns: List[str] | None = None,
    ) -> np.ndarray:
        """Return the 2-D numpy array of training values.

        When *feature_columns* is non-empty the returned array includes
        ``[target] + feature_columns`` with target always at column 0.
        Otherwise returns ``(n, 1)`` — backward compatible.

        Parameters
        ----------
        subset_df : pd.DataFrame
            Subset of the full DataFrame for a single dataset, with a
            ``DatetimeIndex``.
        target_column : str
            Name of the target column (e.g. ``'y'``).
        params : dict
            Strategy parameters.  Checked for ``train_range`` (tuple of
            start/end date strings) or ``train_start_date`` / ``train_split_date``
            fallbacks.
        split_date : str or None
            Default upper bound of the training window when ``train_range``
            is not provided.
        feature_columns : list[str] or None
            Additional columns to include alongside the target.  When ``None``
            or empty, only the target column is returned.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(n_train_rows, 1 + len(feature_columns))``.
        """
        cols = [target_column] + (feature_columns or [])
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
            return subset_df[train_mask][cols].values
        return subset_df[cols].values

    @staticmethod
    def _scale_and_create_sequences(
        data: np.ndarray,
        look_back: int,
        forecast_horizon: int = 1,
        scaler_class=None,
    ) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
        """Scale *data* and build sliding-window sequences.

        Supports multivariate input: *data* may have shape ``(n, k)`` where
        ``k >= 1``.  X windows include **all** columns while y targets use
        only column 0 (the target).

        Parameters
        ----------
        data : np.ndarray
            2-D array of shape ``(n_samples, n_features)`` where column 0 is
            the prediction target and remaining columns are exogenous features
            (e.g. calendar features).
        look_back : int
            Number of past time steps in each input window.
        forecast_horizon : int, default 1
            Number of future steps in the target vector *y*.  When ``<= 1``
            (default) each sample maps to a single next-step target.  When
            ``> 1`` the targets are a contiguous slice of length
            *forecast_horizon* — used by direct multi-step models.

        Returns
        -------
        X : np.ndarray
            3-D array ``(n_windows, look_back, n_features)``.
        y : np.ndarray
            1-D (single-step) or 2-D (multi-step) target array.
        scaler : RobustScaler
            Fitted on **all** columns — used to scale input windows and to
            reconstruct feature rows during recursive prediction.
        target_scaler : RobustScaler
            Fitted on **column 0 only** — used for inverse-transforming
            predictions back to the original target scale.  When
            ``n_features == 1`` this is functionally identical to *scaler*.
        """
        if scaler_class is None:
            scaler_class = RobustScaler
        scaler = scaler_class()
        scaled_data = scaler.fit_transform(data)

        # Separate scaler for inverse-transforming single-column predictions
        target_scaler = scaler_class()
        target_scaler.fit(data[:, 0:1])

        n_cols = data.shape[1]

        X, y = [], []
        if forecast_horizon <= 1:
            for i in range(len(scaled_data) - look_back):
                X.append(scaled_data[i:(i + look_back), :])      # ALL columns
                y.append(scaled_data[i + look_back, 0])           # target only
        else:
            for i in range(len(scaled_data) - look_back - forecast_horizon + 1):
                X.append(scaled_data[i:(i + look_back), :])       # ALL columns
                y.append(scaled_data[(i + look_back):(i + look_back + forecast_horizon), 0])

        X = np.array(X)
        y = np.array(y)
        # X shape is already (n_samples, look_back, n_cols) — no reshape needed
        return X, y, scaler, target_scaler

    # ------------------------------------------------------------------
    # Shared predict helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_model_objects(
        model_objects: Any,
        strategy_name: str,
        logger: logging.Logger,
    ) -> Tuple[Any, RobustScaler, RobustScaler, int, int | None, List[str], dict] | None:
        """Extract model bundle components from *model_objects*.

        Backward compatible: old saved models without ``target_scaler`` or
        ``feature_columns`` fall back to ``scaler`` and ``[]`` respectively.

        Parameters
        ----------
        model_objects : dict or Any
            Model bundle dict produced by :meth:`train`.  Expected keys:
            ``keras_model``, ``scaler``, and optionally ``target_scaler``,
            ``look_back``, ``forecast_horizon``, ``feature_columns``, ``params``.
        strategy_name : str
            Human-readable name used in error messages.
        logger : logging.Logger
            Logger instance for error reporting.

        Returns
        -------
        tuple or None
            ``(model, scaler, target_scaler, look_back, forecast_horizon,
            feature_columns, params)`` on success; ``None`` if the bundle
            format is unrecognised.
        """
        if isinstance(model_objects, dict) and 'keras_model' in model_objects:
            return (
                model_objects['keras_model'],
                model_objects['scaler'],
                model_objects.get('target_scaler', model_objects['scaler']),
                model_objects.get('look_back', 30),
                model_objects.get('forecast_horizon'),
                model_objects.get('feature_columns', []),
                model_objects.get('params', {}),
            )
        logger.error(f"{strategy_name} requires a dict with keras_model and scaler.")
        return None

    @staticmethod
    def _resolve_predict_bounds(params: dict, forecast_horizon: int | None = None) -> Tuple[int, str | None, str | None]:
        """Return ``(forecast_horizon, predict_start_date, predict_end_date)``.

        Resolves the effective prediction date window from *params*, falling
        back to *forecast_horizon* when ``test_range`` is absent.

        Parameters
        ----------
        params : dict
            Strategy/model parameters.  Checked for ``forecast_horizon``,
            ``predict_start_date``, ``predict_end_date``, and ``test_range``.
        forecast_horizon : int or None
            Default horizon when *params* does not specify one.

        Returns
        -------
        tuple[int, str | None, str | None]
            ``(forecast_horizon, predict_start_date, predict_end_date)``.
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

    def _run_one_recursive_origin(
        self,
        seed_seq: np.ndarray,
        model,
        scaler,
        effective_ts,
        forecast_horizon: int,
        feat_cols: List[str],
        step_dates: List,
    ) -> List[float]:
        """Run one recursive origin: h 1-step predictions with autoregressive feedback.

        This is the **shared inner loop** used by both evaluation
        (:meth:`_predict_recursive`) and deployment (:meth:`_predict_schedule`).
        The only difference between those two callers is which seed window and
        which step dates they pass in — the algorithm itself is identical.

        Parameters
        ----------
        seed_seq : np.ndarray
            Shape ``(look_back, n_cols)``, already scaled and clipped.
            The initial context window for this origin.
        model : keras.Model
            Trained model that outputs a single scalar per forward pass.
        scaler :
            Full-column scaler (used to re-scale the reconstructed calendar row).
        effective_ts :
            Target-only scaler used to inverse-transform the scalar prediction
            when reconstructing the calendar-feature row.
        forecast_horizon : int
            Number of steps to predict.
        feat_cols : list[str]
            Exogenous feature column names.  When non-empty,
            :func:`~src.forecast.feature_engineering.compute_calendar_row`
            is called at each step to fill the feature part of the new row.
        step_dates : list
            Sequence of ``pd.Timestamp`` objects, one per step (length =
            *forecast_horizon*).  Used as the date argument to
            ``compute_calendar_row`` when *feat_cols* is non-empty.

        Returns
        -------
        list[float]
            Scaled predictions of length *forecast_horizon*.
            Inverse-transform with *effective_ts* to recover original units.
        """
        current_seq = seed_seq.copy()
        predictions_scaled: List[float] = []
        for step in range(forecast_horizon):
            pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)
            pred_val = float(pred[0, 0])
            predictions_scaled.append(pred_val)
            current_seq = np.roll(current_seq, -1, axis=0)
            if feat_cols:
                from src.forecast.feature_engineering import compute_calendar_row
                pred_unscaled = effective_ts.inverse_transform(
                    np.array([[pred_val]])
                )[0, 0]
                cal_values = compute_calendar_row(step_dates[step])
                unscaled_row = np.array(
                    [[pred_unscaled] + [cal_values[c] for c in feat_cols]]
                )
                scaled_row = np.clip(scaler.transform(unscaled_row), -3.0, 3.0)
                current_seq[-1, :] = scaled_row[0, :]
            else:
                current_seq[-1, 0] = pred_val
        return predictions_scaled

    def _predict_schedule(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_names: List[str],
        model,
        scaler: RobustScaler,
        look_back: int,
        forecast_horizon: int,
        predict_start_date: str | None,
        predict_end_date: str | None,
        target_scaler: RobustScaler | None = None,
        feature_columns: List[str] | None = None,
    ) -> dict:
        """Run **schedule** (deployment) prediction: one origin, future dates.

        Seeds from the last *look_back* observations in *df* and applies the
        shared recursive inner loop (:meth:`_run_one_recursive_origin`) once,
        producing *forecast_horizon* values for dates that lie **beyond** the
        dataset.  Because these future actuals are unknown at inference time,
        no ``actuals`` or ``windows`` are returned — this mode is for live
        operational use (e.g. the API or CLI schedule endpoint), not for
        offline metric evaluation.

        Use :meth:`_predict_recursive` when you need to evaluate accuracy over
        a held-out test set (it applies the same algorithm at every origin in
        the test period and returns ``actuals`` and ``windows``).

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with a ``dataset_name`` column and ``DatetimeIndex``.
        target_column : str
            Name of the target column.
        dataset_names : list[str]
            Datasets to iterate over.
        model : keras.Model
            Trained Keras model with scalar (1-step) output.
        scaler : RobustScaler
            Fitted on all input columns (target + features).
        look_back : int
            Window size expected by the model.
        forecast_horizon : int
            Number of future steps to predict.
        predict_start_date, predict_end_date : str or None
            Optional date bounds to filter the seed data.
        target_scaler : RobustScaler or None
            Scaler fitted on the target column only.  Falls back to *scaler*
            when ``None``.
        feature_columns : list[str] or None
            Exogenous feature columns reconstructed at each step via
            :func:`~src.forecast.feature_engineering.compute_calendar_row`.

        Returns
        -------
        dict
            Keys: ``prediction``, ``values``, ``dates``, ``value``, ``date``,
            ``created_at``.  No ``actuals`` or ``windows`` keys.
        """
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

            effective_ts = target_scaler if target_scaler is not None else scaler
            feat_cols = feature_columns or []
            cols = [target_column] + feat_cols
            data = subset[cols].values[-look_back:]
            scaled_data = np.clip(scaler.transform(data), -3.0, 3.0)
            current_seq = scaled_data.copy()

            # Determine last date for future date range (moved before loop)
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

            predictions_scaled = self._run_one_recursive_origin(
                current_seq, model, scaler, effective_ts,
                forecast_horizon, feat_cols, list(future_dates),
            )
            predictions_series = effective_ts.inverse_transform(
                np.array(predictions_scaled).reshape(-1, 1)
            ).flatten().tolist()

            output_dict.update({
                self.output_key: predictions_series,
                'values': predictions_series,
                'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'value': predictions_series[0] if predictions_series else 0,
                'date': datetime.now(),
                'created_at': datetime.now(),
            })

        return output_dict

    def _predict_recursive(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_names: List[str],
        model,
        scaler: RobustScaler,
        look_back: int,
        forecast_horizon: int,
        predict_start_date: str | None,
        predict_end_date: str | None,
        target_scaler: RobustScaler | None = None,
        feature_columns: List[str] | None = None,
    ) -> dict:
        """Run **recursive rolling-origin** H-step evaluation (stride=1).

        For each origin ``t`` in ``[look_back, n - forecast_horizon]``, the model
        is seeded with the **actual** context window ``[t-look_back : t]`` and then
        runs ``forecast_horizon`` recursive 1-step predictions, feeding its own
        output back at each step.  This is the academically correct multi-horizon
        evaluation: the model is never aware of actuals beyond its seed.

        Returns a flat ``(values, actuals, dates)`` suitable for metric
        computation and a ``windows`` list of per-origin dicts for plotting.

        Parameters
        ----------
        df : pd.DataFrame
            Full test dataset (including look_back context rows at the start)
            with ``dataset_name`` column and ``DatetimeIndex``.
        target_column : str
            Name of the target column.
        dataset_names : list[str]
            Datasets to iterate over.
        model : keras.Model
            Trained Keras model with scalar output (1 step per forward pass).
        scaler : RobustScaler
            Fitted on all input columns.
        look_back : int
            Window size expected by the model.
        forecast_horizon : int
            Number of future steps to predict per origin.
        predict_start_date, predict_end_date : str or None
            Optional date bounds to filter ``df``.
        target_scaler : RobustScaler or None
            Scaler fitted on the target column only.
        feature_columns : list[str] or None
            Exogenous feature columns present in the input window.

        Returns
        -------
        dict
            Keys: ``prediction``, ``values``, ``actuals``, ``dates``,
            ``windows``, ``value``, ``date``, ``created_at``.
        """
        output_dict: Dict[str, Any] = {}

        for dataset_name in dataset_names:
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

            effective_ts = target_scaler if target_scaler is not None else scaler
            feat_cols = feature_columns or []
            cols = [target_column] + feat_cols
            all_data = subset[cols].values
            scaled_all = np.clip(scaler.transform(all_data), -3.0, 3.0)

            n = len(scaled_all)
            if n < look_back + forecast_horizon:
                self.logger.warning(
                    f"Not enough data for {dataset_name} to run recursive evaluation "
                    f"with look_back={look_back}, forecast_horizon={forecast_horizon} "
                    f"(n={n})"
                )
                continue

            has_dates = isinstance(subset.index, pd.DatetimeIndex)
            all_values: List[float] = []
            all_actuals: List[float] = []
            all_dates: List[str] = []
            windows: List[dict] = []

            # Origins: look_back .. n-forecast_horizon (inclusive both ends)
            for t in range(look_back, n - forecast_horizon + 1):
                current_seq = scaled_all[t - look_back:t].copy()  # (look_back, n_cols)

                if has_dates:
                    step_dates = [subset.index[t] + pd.Timedelta(days=s) for s in range(forecast_horizon)]
                else:
                    step_dates = [pd.Timestamp.now()] * forecast_horizon
                predictions_scaled = self._run_one_recursive_origin(
                    current_seq, model, scaler, effective_ts,
                    forecast_horizon, feat_cols, step_dates,
                )

                # Inverse-scale predictions (RobustScaler only — log/diff undone by pipeline)
                preds_unscaled = effective_ts.inverse_transform(
                    np.array(predictions_scaled).reshape(-1, 1)
                ).flatten()

                # Actuals for this window (same scale as predictions)
                actuals_unscaled = effective_ts.inverse_transform(
                    scaled_all[t:t + forecast_horizon, 0:1]
                ).flatten()

                if has_dates:
                    future_dates_idx = subset.index[t:t + forecast_horizon]
                    origin_date_str = subset.index[t - 1].strftime('%Y-%m-%d')
                    future_dates_str = future_dates_idx.strftime('%Y-%m-%d').tolist()
                else:
                    future_dates_str = [str(t + s) for s in range(forecast_horizon)]
                    origin_date_str = str(t - 1)

                all_values.extend(preds_unscaled.tolist())
                all_actuals.extend(actuals_unscaled.tolist())
                all_dates.extend(future_dates_str)
                windows.append({
                    'origin_date': origin_date_str,
                    'future_dates': future_dates_str,
                    'predictions': preds_unscaled.tolist(),
                    'actuals': actuals_unscaled.tolist(),
                })

            if not all_values:
                self.logger.warning(
                    f"_predict_recursive produced no predictions for {dataset_name}"
                )
                continue

            output_dict.update({
                self.output_key: all_values,
                'values': all_values,
                'actuals': all_actuals,
                'dates': all_dates,
                'windows': windows,
                'value': all_values[0] if all_values else 0,
                'date': datetime.now(),
                'created_at': datetime.now(),
            })

        if not output_dict:
            raise ValueError(
                f"_predict_recursive produced no predictions: all datasets were empty "
                f"or had fewer rows than look_back={look_back}+forecast_horizon={forecast_horizon}. "
                f"Datasets: {dataset_names}"
            )

        return output_dict

    def _predict_mimo(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_names: List[str],
        model,
        scaler: RobustScaler,
        look_back: int,
        forecast_horizon: int,
        predict_start_date: str | None,
        predict_end_date: str | None,
        target_scaler: RobustScaler | None = None,
        feature_columns: List[str] | None = None,
    ) -> dict:
        """Run **MIMO rolling-origin** H-step evaluation (stride=1).

        For each origin ``t`` in ``[look_back, n - forecast_horizon]``, a single
        forward pass through the model (which must have a ``Dense(forecast_horizon)``
        output layer) produces all ``forecast_horizon`` predictions simultaneously.

        Returns the same structure as :meth:`_predict_recursive` for consistency.

        Parameters
        ----------
        df : pd.DataFrame
            Full test dataset with ``dataset_name`` column and ``DatetimeIndex``.
        target_column : str
            Name of the target column.
        dataset_names : list[str]
            Datasets to iterate over.
        model : keras.Model
            Trained Keras model whose output shape is ``(batch, forecast_horizon)``.
        scaler : RobustScaler
            Fitted on all input columns.
        look_back : int
            Window size expected by the model.
        forecast_horizon : int
            Number of future steps the model outputs per forward pass.
        predict_start_date, predict_end_date : str or None
            Optional date bounds to filter ``df``.
        target_scaler : RobustScaler or None
            Scaler fitted on the target column only.
        feature_columns : list[str] or None
            Exogenous feature columns present in the input window.

        Returns
        -------
        dict
            Keys: ``prediction``, ``values``, ``actuals``, ``dates``,
            ``windows``, ``value``, ``date``, ``created_at``.
        """
        output_dict: Dict[str, Any] = {}

        for dataset_name in dataset_names:
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

            effective_ts = target_scaler if target_scaler is not None else scaler
            feat_cols = feature_columns or []
            cols = [target_column] + feat_cols
            all_data = subset[cols].values
            scaled_all = np.clip(scaler.transform(all_data), -3.0, 3.0)

            n = len(scaled_all)
            if n < look_back + forecast_horizon:
                self.logger.warning(
                    f"Not enough data for {dataset_name} to run MIMO evaluation "
                    f"with look_back={look_back}, forecast_horizon={forecast_horizon} "
                    f"(n={n})"
                )
                continue

            has_dates = isinstance(subset.index, pd.DatetimeIndex)
            all_values: List[float] = []
            all_actuals: List[float] = []
            all_dates: List[str] = []
            windows: List[dict] = []

            for t in range(look_back, n - forecast_horizon + 1):
                seed = scaled_all[t - look_back:t][np.newaxis, :, :]  # (1, look_back, n_cols)

                # Single forward pass → (1, forecast_horizon)
                pred_scaled = model.predict(seed, verbose=0)
                preds_unscaled = effective_ts.inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).flatten()

                actuals_unscaled = effective_ts.inverse_transform(
                    scaled_all[t:t + forecast_horizon, 0:1]
                ).flatten()

                if has_dates:
                    future_dates_idx = subset.index[t:t + forecast_horizon]
                    origin_date_str = subset.index[t - 1].strftime('%Y-%m-%d')
                    future_dates_str = future_dates_idx.strftime('%Y-%m-%d').tolist()
                else:
                    future_dates_str = [str(t + s) for s in range(forecast_horizon)]
                    origin_date_str = str(t - 1)

                all_values.extend(preds_unscaled.tolist())
                all_actuals.extend(actuals_unscaled.tolist())
                all_dates.extend(future_dates_str)
                windows.append({
                    'origin_date': origin_date_str,
                    'future_dates': future_dates_str,
                    'predictions': preds_unscaled.tolist(),
                    'actuals': actuals_unscaled.tolist(),
                })

            if not all_values:
                self.logger.warning(
                    f"_predict_mimo produced no predictions for {dataset_name}"
                )
                continue

            output_dict.update({
                self.output_key: all_values,
                'values': all_values,
                'actuals': all_actuals,
                'dates': all_dates,
                'windows': windows,
                'value': all_values[0] if all_values else 0,
                'date': datetime.now(),
                'created_at': datetime.now(),
            })

        if not output_dict:
            raise ValueError(
                f"_predict_mimo produced no predictions: all datasets were empty "
                f"or had fewer rows than look_back={look_back}+forecast_horizon={forecast_horizon}. "
                f"Datasets: {dataset_names}"
            )

        return output_dict

    def _predict_direct_multistep(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_names: List[str],
        model,
        scaler: RobustScaler,
        look_back: int,
        forecast_horizon: int,
        predict_start_date: str | None,
        predict_end_date: str | None,
        target_scaler: RobustScaler | None = None,
        feature_columns: List[str] | None = None,
    ) -> dict:
        """Run **direct multi-step** prediction for each dataset.

        A single forward pass produces all *forecast_horizon* values at once.
        Requires the model's output layer to be ``Dense(forecast_horizon)``.

        Unlike :meth:`_predict_schedule`, no recursive feedback loop is used;
        the model directly outputs the full prediction vector from the last
        *look_back* observations.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with ``dataset_name`` column and ``DatetimeIndex``.
        target_column : str
            Name of the target column.
        dataset_names : list[str]
            Datasets to iterate over.
        model : keras.Model
            Trained Keras model whose output shape is ``(batch, forecast_horizon)``.
        scaler : RobustScaler
            Fitted on all input columns.
        look_back : int
            Window size expected by the model.
        forecast_horizon : int
            Number of future steps the model outputs in a single pass.
        predict_start_date, predict_end_date : str or None
            Optional date bounds to filter the seed data.
        target_scaler : RobustScaler or None
            Scaler fitted on the target column only.
        feature_columns : list[str] or None
            Exogenous feature columns present in the input window.

        Returns
        -------
        dict
            Keys: ``prediction``, ``values``, ``dates``, ``value``, ``date``,
            ``created_at``.
        """
        output_dict: Dict[str, Any] = {}

        for dataset_name in dataset_names:
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
                self.logger.warning(
                    f"Not enough data for {dataset_name} to fulfill look_back of {look_back}"
                )
                continue

            # Take the last look_back values as input seed
            effective_ts = target_scaler if target_scaler is not None else scaler
            feat_cols = feature_columns or []
            cols = [target_column] + feat_cols
            data = subset[cols].values[-look_back:]
            scaled_data = np.clip(scaler.transform(data), -3.0, 3.0)

            # Single forward pass → (1, forecast_horizon) output
            pred_scaled = model.predict(
                scaled_data[np.newaxis, :, :], verbose=0,
            )
            # pred_scaled shape: (1, forecast_horizon)
            predictions_unscaled = effective_ts.inverse_transform(
                pred_scaled.reshape(-1, 1),
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
        scaler: RobustScaler,
        look_back: int,
        target_scaler: RobustScaler | None = None,
        feature_columns: List[str] | None = None,
    ) -> dict:
        """Run **backtest** (rolling one-step) prediction for each dataset.

        For each time step from ``look_back`` to the end of the dataset, the
        model predicts one step ahead using **actual** recent data as context
        (not its own past predictions).  This produces aligned prediction and
        actual arrays suitable for metric computation.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with ``dataset_name`` column and ``DatetimeIndex``.
        target_column : str
            Name of the target column.
        dataset_names : list[str]
            Datasets to iterate over.
        model : keras.Model
            Trained Keras model.
        scaler : RobustScaler
            Fitted on all input columns.
        look_back : int
            Window size expected by the model.
        target_scaler : RobustScaler or None
            Scaler fitted on the target column only.
        feature_columns : list[str] or None
            Exogenous feature columns present in the input window.

        Returns
        -------
        dict
            Keys: ``prediction``, ``values``, ``actuals``, ``dates``,
            ``value``, ``date``, ``created_at``.
        """
        output_dict: Dict[str, Any] = {}

        for dataset_name in dataset_names:
            subset = df.loc[df['dataset_name'] == dataset_name].copy()
            if subset.empty:
                continue

            effective_ts = target_scaler if target_scaler is not None else scaler
            feat_cols = feature_columns or []
            cols = [target_column] + feat_cols
            all_data = subset[cols].values
            scaled_all = np.clip(scaler.transform(all_data), -3.0, 3.0)

            if len(scaled_all) <= look_back:
                self.logger.warning(
                    f"Not enough data for {dataset_name} to backtest with look_back={look_back}"
                )
                continue

            predictions_scaled = []
            for i in range(look_back, len(scaled_all)):
                seq = scaled_all[i - look_back:i][np.newaxis, :, :]
                pred = model.predict(seq, verbose=0)
                predictions_scaled.append(pred[0, 0])

            predictions_unscaled = effective_ts.inverse_transform(
                np.array(predictions_scaled).reshape(-1, 1)
            ).flatten().tolist()

            actuals_unscaled = all_data[look_back:, 0].flatten().tolist()

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

        if not output_dict:
            raise ValueError(
                f"_predict_backtest produced no predictions: all datasets were empty "
                f"or had fewer rows than look_back={look_back}. "
                f"Datasets: {dataset_names}"
            )

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

            # Direct multi-step (MIMO): model outputs forecast_horizon values at once.
            # Recursive mode trains with output_steps=1 (1-step model used in rolling fashion).
            predict_mode = strategy_params.get('predict_mode')
            output_steps = (
                forecast_horizon
                if predict_mode in ('direct_multistep', 'mimo') and forecast_horizon
                else 1
            )
            strategy_params['output_steps'] = output_steps

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
                effective_feature_cols = feature_columns if feature_columns else []
                data = self._filter_train_data(
                    subset_df, target_column, strategy_params, split_date,
                    feature_columns=effective_feature_cols,
                )
                X, y, scaler, target_scaler = self._scale_and_create_sequences(
                    data, look_back, forecast_horizon=output_steps,
                    scaler_class=self._scaler_class,
                )

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
                        'target_scaler': target_scaler,
                        'look_back': look_back,
                        'forecast_horizon': forecast_horizon,
                        'feature_columns': effective_feature_cols,
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

        # Optional Keras callbacks.  When validation_split > 0 the monitor
        # switches to 'val_loss' so EarlyStopping/ReduceLROnPlateau see
        # held-out data and cannot over-fit to training loss.
        validation_split = strategy_params.get('validation_split', 0.0)
        monitor = 'val_loss' if validation_split > 0 else 'loss'
        callbacks = []
        if strategy_params.get('use_lr_scheduler', False):
            callbacks.append(ReduceLROnPlateau(
                monitor=monitor, patience=10, factor=0.5, min_lr=1e-6, verbose=0,
            ))
        if strategy_params.get('use_early_stopping', False):
            callbacks.append(EarlyStopping(
                monitor=monitor, patience=20, restore_best_weights=True, verbose=0,
            ))

        self.logger.info(
            f"Training {self._strategy_display_name} model for {epochs} epochs, "
            f"batch size {batch_size}..."
        )
        model.fit(
            X, y, epochs=epochs, batch_size=batch_size, verbose=0,
            validation_split=validation_split,
            callbacks=callbacks if callbacks else None,
        )
        return model

    def _finetune_on_window(
        self,
        model,
        data: np.ndarray,
        look_back: int,
        strategy_params: dict,
    ) -> Tuple[Any, RobustScaler, RobustScaler]:
        """Fine-tune *model* on a recent sliding window without rebuilding the architecture.

        Used by the rolling-training backtest path
        (``use_rolling_training=True, rolling_retrain_mode='finetune'``).

        A **new** pair of :class:`~sklearn.preprocessing.RobustScaler` objects
        is fitted on *data* so that the scaler always reflects the current
        demand regime.  The existing model weights are updated in-place by
        running a short ``model.fit()`` call — ``build_model()`` is **not**
        called, preserving the learned representations while adapting to the
        new distribution.

        Parameters
        ----------
        model : keras.Model
            Existing trained Keras model whose weights will be fine-tuned.
        data : np.ndarray
            2-D array of shape ``(n_window, n_features)`` in the **original
            (un-scaled) space**.  Column 0 must be the target.
        look_back : int
            Sliding-window size used to build sequences.
        strategy_params : dict
            Strategy hyperparameters.  Read keys: ``rolling_finetune_epochs``
            (default 10), ``batch_size`` (default 32).

        Returns
        -------
        (model, new_scaler, new_target_scaler)
            The fine-tuned Keras model and the two freshly fitted scalers to
            replace the stale ones in the model bundle.
        """
        finetune_epochs = strategy_params.get('rolling_finetune_epochs', 10)
        batch_size = strategy_params.get('batch_size', 32)

        X, y, new_scaler, new_target_scaler = self._scale_and_create_sequences(
            data, look_back, forecast_horizon=1,
            scaler_class=self._scaler_class,
        )
        if len(X) == 0:
            self.logger.warning(
                "_finetune_on_window: window too small to build sequences "
                "(n_window=%d, look_back=%d) — skipping fine-tune step.",
                len(data), look_back,
            )
            return model, new_scaler, new_target_scaler

        self.logger.info(
            "Fine-tuning %s for %d epochs on %d-day window (%d sequences).",
            self._strategy_display_name, finetune_epochs, len(data), len(X),
        )
        model.fit(X, y, epochs=finetune_epochs, batch_size=batch_size, verbose=0)
        return model, new_scaler, new_target_scaler

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
            model, scaler, target_scaler, look_back, forecast_horizon, feature_columns_stored, params = unpacked

            forecast_horizon, predict_start_date, predict_end_date = (
                self._resolve_predict_bounds(params, forecast_horizon)
            )

            if submode == 'recursive':
                result = self._predict_recursive(
                    df, target_column, dataset_names,
                    model, scaler, look_back,
                    forecast_horizon, predict_start_date, predict_end_date,
                    target_scaler=target_scaler,
                    feature_columns=feature_columns_stored,
                )
            elif submode == 'mimo':
                result = self._predict_mimo(
                    df, target_column, dataset_names,
                    model, scaler, look_back,
                    forecast_horizon, predict_start_date, predict_end_date,
                    target_scaler=target_scaler,
                    feature_columns=feature_columns_stored,
                )
            elif submode == 'schedule':
                result = self._predict_schedule(
                    df, target_column, dataset_names,
                    model, scaler, look_back,
                    forecast_horizon, predict_start_date, predict_end_date,
                    target_scaler=target_scaler,
                    feature_columns=feature_columns_stored,
                )
            elif submode == 'direct_multistep':
                result = self._predict_direct_multistep(
                    df, target_column, dataset_names,
                    model, scaler, look_back,
                    forecast_horizon, predict_start_date, predict_end_date,
                    target_scaler=target_scaler,
                    feature_columns=feature_columns_stored,
                )
            else:
                result = self._predict_backtest(
                    df, target_column, dataset_names,
                    model, scaler, look_back,
                    target_scaler=target_scaler,
                    feature_columns=feature_columns_stored,
                )

            output_dict['predict'].update(result)

        except Exception as e:
            self.logger.error(
                f"{self._strategy_display_name} predict failed: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

        return output_dict
