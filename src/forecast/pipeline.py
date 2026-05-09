"""
Forecast pipeline orchestrator.

``run_forecast_pipeline()`` is the central entry point that ties together:

1. **Data fetching** — via ``src.data.data_fetcher``
2. **Feature engineering** — via :mod:`src.forecast.feature_engineering`
3. **Model training** — delegated to the appropriate ``ModelStrategy``
4. **Prediction + evaluation** — metrics computed on the original (untransformed) scale
5. **Visualisation** — train/test split and actual-vs-predicted plots
6. **MLflow logging** — optional experiment tracking

The pipeline itself does **not** implement model architectures or feature
construction logic; those responsibilities belong to the strategy classes
and :mod:`feature_engineering` respectively.

Target transform round-trip
---------------------------
Three orthogonal boolean flags in ``strategy_params`` control optional
pre-processing applied to **all** strategies uniformly:

* ``use_log_transform`` — compresses the target range via ``log1p``
* ``use_differencing`` — makes the target stationary via first-order diff
* ``use_calendar_features`` — enriches the DataFrame with temporal features

Forward transforms are applied **before** the train/test split so that
every strategy sees the same transformed ``'y'`` column.  After prediction,
the pipeline inverts the transforms **before** computing metrics, so all
reported numbers (MSE, RMSE, MAE, MAPE, SMAPE) are in the **original kWh
scale**.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, cast, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from src.forecast.feature_engineering import (
    add_calendar_features,
    add_tree_lag_features,
    apply_forward_transforms,
    apply_inverse_transforms,
)
from src.forecast.model_persistence import save_model
from src.forecast.strategies import (
    HussainHybridModelStrategy,
    HussainTransformerModelStrategy,
    HussainLSTMModelStrategy,
    HybridTransformerLSTMModelStrategy,
    LightGBMModelStrategy,
    LSTMModelStrategy,
    TransformerModelStrategy,
    XGBoostModelStrategy,
)
from src.forecast.strategies.utils_ts import mase, smape
from src.data.data_fetcher import fetch_daily_energy_for_forecast
from src.analysis.forecast_plots import plot_train_test_split, plot_test_vs_predict

# MLflow (optional)
try:
    import mlflow
    import joblib
    import tempfile
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

# Suppress noisy matplotlib logger
logging.getLogger('matplotlib').setLevel(logging.WARNING)


# ── Strategy factory ───────────────────────────────────────────────────────

# Maps model_type → (StrategyClass, params_attr_key)
_STRATEGY_REGISTRY: Dict[str, tuple] = {
    'transformer':          (TransformerModelStrategy,              'transformer_params'),
    'lightgbm':             (LightGBMModelStrategy,                 'lgbm_params'),
    'xgboost':              (XGBoostModelStrategy,                  'xgb_params'),
    'hybrid':               (HybridTransformerLSTMModelStrategy,    'hybrid_params'),
    'hussain_lstm':         (HussainLSTMModelStrategy,              'lstm_params'),
    'hussain_transformer':  (HussainTransformerModelStrategy,       'hussain_transformer_params'),
    'hussain_hybrid':       (HussainHybridModelStrategy,            'hussain_hybrid_params'),
    'lstm':                 (LSTMModelStrategy,                     'lstm_params'),
}


def _make_strategy(model_type: str):
    """Return ``(strategy_instance, params_attr_key)`` for *model_type*."""
    entry = _STRATEGY_REGISTRY.get(model_type)
    if entry is None:
        # Default fallback to LSTM
        entry = _STRATEGY_REGISTRY['lstm']
    cls, attr_key = entry
    return cls(), attr_key


# ── Metrics persistence ────────────────────────────────────────────────────

def _build_metrics_record(
    experiment_name: str,
    model_type: str,
    dataset: str,
    forecast_horizon: int,
    split_date_str: str,
    strategy_params: dict,
    metrics: dict,
) -> dict:
    """Flatten one experiment run into a single row dict for the metrics CSV.

    Scalar ``strategy_params`` values are stored inline under ``param_<key>``;
    complex values (lists, dicts) are JSON-encoded strings so the CSV stays
    flat and human-readable.
    """
    record: dict = {
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'experiment_name': experiment_name,
        'model_type': model_type,
        'dataset': dataset,
        'forecast_horizon': forecast_horizon,
        'split_date': split_date_str,
    }
    for k, v in strategy_params.items():
        if k == 'forecast_horizon':
            continue  # already captured above
        if isinstance(v, (int, float, str, bool)) or v is None:
            record[f'param_{k}'] = v
        else:
            record[f'param_{k}'] = json.dumps(v)
    record.update(metrics)
    return record


def _upsert_metrics_csv(
    records: List[dict],
    output_dir: str = 'output_metrics',
    filename: str = 'forecast_metrics.csv',
) -> str:
    """Persist experiment metrics to a CSV used as a running database.

    Records are keyed on ``(experiment_name, dataset, forecast_horizon)``.
    Existing rows that match an incoming key are *replaced*; all other rows
    are kept intact, so no previous run is ever silently discarded.

    Returns the path to the written CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)

    new_df = pd.DataFrame(records)
    key_cols = ['experiment_name', 'dataset', 'forecast_horizon']

    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Could not read existing metrics CSV (%s); starting fresh.", exc)
            existing_df = pd.DataFrame()

        if not existing_df.empty and all(c in existing_df.columns for c in key_cols):
            new_keys = set(new_df[key_cols].apply(tuple, axis=1))
            keep_mask = ~existing_df[key_cols].apply(tuple, axis=1).isin(new_keys)
            combined_df = pd.concat([existing_df[keep_mask], new_df], ignore_index=True)
        else:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(csv_path, index=False)
    logger.info("Metrics upserted → %s (%d total rows)", csv_path, len(combined_df))
    return csv_path


# ── Rolling backtest ───────────────────────────────────────────────────────

def _run_rolling_backtest(
    df_full: pd.DataFrame,
    dataset: str,
    strategy,
    strategy_params: dict,
    params_key: str,
    feature_cols: List[str],
    look_back: int,
    split_date_str: str,
    model_name_prefix: str,
) -> tuple:
    """Rolling-retrain backtest for neural models.

    Iterates over the test set in batches of ``rolling_retrain_interval`` days.
    At the start of each batch the model is either fully retrained
    (``rolling_retrain_mode='full'``) or fine-tuned
    (``rolling_retrain_mode='finetune'``) on the most recent
    ``rolling_window_days`` of **pre-batch** history.  A fresh
    :class:`~sklearn.preprocessing.RobustScaler` is fitted on that window at
    every retrain step so that the scaler always reflects the current demand
    regime.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full dataset with ``dataset_name`` column and ``DatetimeIndex``.
        Should include both training and test rows (transformed, with calendar
        features already added where applicable).
    dataset : str
        Dataset name used to filter ``df_full``.
    strategy : ModelStrategy
        Instantiated strategy object (e.g. ``HybridTransformerLSTMModelStrategy``).
    strategy_params : dict
        Full strategy-params dict.  Read keys: ``rolling_window_days``
        (default 180), ``rolling_retrain_interval`` (default 7),
        ``rolling_retrain_mode`` (default ``'full'``),
        ``rolling_finetune_epochs`` (default 10).
    params_key : str
        The ``df.attrs`` key for this strategy (e.g. ``'hybrid_params'``).
    feature_cols : list[str]
        Exogenous feature columns (empty list for univariate models).
    look_back : int
        Context window size expected by the model.
    split_date_str : str
        Train/test split date string (``'YYYY-MM-DD'``).
    model_name_prefix : str
        Prefix used to locate the trained model in the output dict.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]
        ``(predictions, actuals, dates)`` — all in **transformed** space
        (inverse transforms are applied by the caller, i.e.
        ``run_forecast_pipeline``).
    """
    rolling_window_days = strategy_params.get('rolling_window_days', 180)
    retrain_interval = strategy_params.get('rolling_retrain_interval', 7)
    retrain_mode = strategy_params.get('rolling_retrain_mode', 'full')

    min_window = 5 * look_back
    if rolling_window_days < min_window:
        logger.warning(
            "_run_rolling_backtest: rolling_window_days=%d is below 5×look_back=%d. "
            "Clamping to %d to ensure enough sequences for training.",
            rolling_window_days, min_window, min_window,
        )
        rolling_window_days = min_window

    split_date = pd.to_datetime(split_date_str)
    subset = df_full.loc[df_full['dataset_name'] == dataset].copy()
    test_idx = subset.index[subset.index > split_date]

    if len(test_idx) == 0:
        logger.warning("_run_rolling_backtest: no test rows for %s after %s.", dataset, split_date_str)
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    all_predictions: List[float] = []
    all_actuals: List[float] = []
    all_dates: List[Any] = []

    model_objects: Any = None  # populated on first retrain

    # Iterate over test days in batches of retrain_interval
    batch_starts = range(0, len(test_idx), retrain_interval)
    cols = ['y'] + feature_cols

    for batch_idx, batch_start in enumerate(batch_starts):
        batch_dates = test_idx[batch_start: batch_start + retrain_interval]
        batch_anchor = batch_dates[0]  # first day of this batch

        # Build the training window: all rows strictly before batch_anchor
        history = subset.loc[subset.index < batch_anchor]
        if len(history) < rolling_window_days:
            logger.warning(
                "_run_rolling_backtest [%s] batch %d: only %d history rows available "
                "(need %d). Using all available.",
                dataset, batch_idx, len(history), rolling_window_days,
            )
            window_df = history.copy()
        else:
            window_df = history.iloc[-rolling_window_days:].copy()

        if len(window_df) <= look_back:
            logger.warning(
                "_run_rolling_backtest [%s] batch %d: window_df (%d rows) ≤ look_back (%d). "
                "Skipping retrain.",
                dataset, batch_idx, len(window_df), look_back,
            )
        else:
            # ── Retrain ────────────────────────────────────────────────
            if retrain_mode == 'finetune' and model_objects is not None:
                # Fine-tune existing weights + refit scalers on current window
                window_data = window_df[cols].values
                model_objects['keras_model'], model_objects['scaler'], model_objects['target_scaler'] = (
                    strategy._finetune_on_window(
                        model_objects['keras_model'],
                        window_data,
                        look_back,
                        strategy_params,
                    )
                )
            else:
                # Full retrain from scratch on the current window.
                # Free the previous model from TF's memory before building a
                # new graph — otherwise each iteration's model accumulates in
                # TF's session and eventually triggers an OOM kill.
                if model_objects is not None:
                    try:
                        import keras.backend as _K
                        del model_objects['keras_model']
                        _K.clear_session()
                        gc.collect()
                    except Exception:
                        pass
                    model_objects = cast(Any, None)  # reset; next retrain will assign a fresh bundle
                train_params = {k: v for k, v in strategy_params.items()
                                if k not in ('train_range', 'train_split_date')}
                window_df.attrs[params_key] = train_params
                trained_dict = strategy.train(
                    df=window_df,
                    feature_columns=feature_cols,
                    target_column='y',
                    dataset_names=[dataset],
                    model_name_prefix=model_name_prefix,
                    split_date=None,  # train on ALL window rows
                )
                key = f"{model_name_prefix}_{dataset}"
                if key not in trained_dict.get('train', {}):
                    logger.warning(
                        "_run_rolling_backtest [%s] batch %d: training produced no model. Skipping.",
                        dataset, batch_idx,
                    )
                    continue
                model_objects = trained_dict['train'][key]['model']

        if model_objects is None:
            logger.warning(
                "_run_rolling_backtest [%s] batch %d: no model available yet. Skipping.",
                dataset, batch_idx,
            )
            continue

        # ── Predict on batch: prepend look_back context rows ─────────
        context_rows = subset.loc[subset.index < batch_anchor].iloc[-look_back:]
        batch_rows = subset.loc[batch_dates]
        predict_df = pd.concat([context_rows, batch_rows])
        predict_df = predict_df.copy()
        # Ensure dataset_name column is present for strategy.predict
        if 'dataset_name' not in predict_df.columns:
            predict_df['dataset_name'] = dataset

        predict_dict = strategy.predict(
            df=predict_df,
            feature_columns=feature_cols,
            target_column='y',
            model_objects=model_objects,
            context_date=None,
            dataset_names=[dataset],
            submode=None,  # backtest mode
        )
        inner = predict_dict.get('predict', {})
        if 'values' not in inner:
            logger.warning(
                "_run_rolling_backtest [%s] batch %d: predict returned no values. Skipping.",
                dataset, batch_idx,
            )
            continue

        batch_preds = inner['values']        # length = len(batch_dates) (look_back warm-up excluded)
        batch_acts = inner.get('actuals', [])
        batch_date_strs = inner.get('dates', [])

        n = min(len(batch_preds), len(batch_acts), len(batch_date_strs), len(batch_dates))
        all_predictions.extend(batch_preds[:n])
        all_actuals.extend(batch_acts[:n])
        all_dates.extend(pd.to_datetime(batch_date_strs[:n]))

    return (
        np.array(all_predictions),
        np.array(all_actuals),
        pd.DatetimeIndex(all_dates),
    )


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_forecast_pipeline(
    datasets: List[str],
    strategy_params: dict,
    split_date_str: str,
    model_type: str = "lstm",
    model_suffix: str = "",
    mlflow_tracking_uri: str | None = None,
    output_dir: str | None = None,
) -> Dict[int, dict]:
    """Execute the full forecast pipeline for one or more datasets.

    This is the **single entry point** called by every ``execute_*`` function
    in ``tfm_forecast.py``.  The pipeline:

    1. Fetches daily energy data
    2. Applies optional forward target transforms (log / differencing)
    3. Adds optional calendar features
    4. For each ``forecast_horizon``:

       a. Splits into train / test
       b. Instantiates the appropriate strategy and prepares features
       c. Trains, predicts, computes metrics on the **original** scale
       d. Produces plots and (optionally) logs to MLflow
    """
    models_dir = 'output_models'
    os.makedirs(models_dir, exist_ok=True)
    li_forecast_horizons = strategy_params.get('li_forecast_horizons', [1])
    plots_dir = output_dir if output_dir else 'output_plots'

    results_summary: List[dict] = []
    horizon_metrics: Dict[int, dict] = {}

    for dataset in datasets:
        logger.info("--- Processing Dataset: %s ---", dataset)

        df = fetch_daily_energy_for_forecast(dataset)
        if df.empty:
            logger.warning("No data found for %s. Skipping.", dataset)
            continue

        # ── Drop excluded date range (e.g. COVID lockdown gap) ─────────
        exclude_range = strategy_params.get('exclude_range')
        if exclude_range:
            excl_start_dt = pd.to_datetime(exclude_range[0]) if exclude_range[0] else None
            excl_end_dt   = pd.to_datetime(exclude_range[1]) if exclude_range[1] else None
            drop_mask = pd.Series(True, index=df.index)
            if excl_start_dt is not None:
                drop_mask &= df.index >= excl_start_dt
            if excl_end_dt is not None:
                drop_mask &= df.index <= excl_end_dt
            n_dropped = int(drop_mask.sum())
            if n_dropped > 0:
                df = df[~drop_mask].copy()
                logger.info(
                    "exclude_range (%s → %s): removed %d rows from '%s'.",
                    exclude_range[0], exclude_range[1], n_dropped, dataset,
                )

        # ── Forward target transforms ──────────────────────────────────
        # Save original (un-transformed) 'y' column for the train/test split
        # plots — forward transforms (e.g. RevIN) must not affect the visual.
        df_plot_original = df[['y']].copy()
        df, transform_state = apply_forward_transforms(df, strategy_params)

        # ── Calendar features (optional, all model types) ──────────────
        use_calendar_features = strategy_params.get('use_calendar_features', False)
        calendar_cols: List[str] = []
        if use_calendar_features:
            df, calendar_cols = add_calendar_features(df)
            logger.info("Added calendar features: %s", calendar_cols)

        for forecast_horizon in li_forecast_horizons:
            logger.info("Prediction window: %d days", forecast_horizon)
            strategy_params['forecast_horizon'] = forecast_horizon

            # 1. Split train/test ───────────────────────────────────────
            if len(df) <= forecast_horizon:
                logger.warning("Not enough data for test size %d. Skipping.", forecast_horizon)
                continue

            split_date = pd.to_datetime(split_date_str)

            test_range = strategy_params.get('test_range')
            if test_range:
                test_start, test_end = test_range
                test_mask = pd.Series(True, index=df.index)
                if test_start:
                    test_mask &= df.index >= pd.to_datetime(test_start)
                else:
                    test_mask &= df.index > split_date
                if test_end:
                    test_mask &= df.index <= pd.to_datetime(test_end)
                test_df = df[test_mask].copy()
            else:
                test_df = df[df.index > split_date].copy()

            train_range = strategy_params.get('train_range')
            if train_range:
                train_start, train_end = train_range
                train_mask = pd.Series(True, index=df.index)
                if train_start:
                    train_mask &= df.index >= pd.to_datetime(train_start)
                if train_end:
                    train_mask &= df.index <= pd.to_datetime(train_end)
                else:
                    train_mask &= df.index <= split_date
                train_df = df[train_mask].copy()
            else:
                train_df = df[df.index <= split_date].copy()

            # 2. Strategy & feature preparation ─────────────────────────
            strategy, params_key = _make_strategy(model_type)
            model_name_prefix = f"{model_type}_{forecast_horizon}d"

            df_model = df.copy()
            df_model['dataset_name'] = dataset
            df_model.attrs[params_key] = strategy_params

            train_df['dataset_name'] = dataset
            train_df.attrs[params_key] = strategy_params

            # Unmodified copies for plotting in original (un-transformed) scale.
            # Use df_plot_original so that forward transforms (e.g. RevIN) do
            # not distort the train/test split visualisation.
            train_df_plot = df_plot_original.loc[
                df_plot_original.index.isin(train_df.index)
            ].copy()
            test_df_plot = df_plot_original.loc[
                df_plot_original.index.isin(test_df.index)
            ].copy()

            feature_cols: List[str] = []
            if model_type in ('xgboost', 'lightgbm'):
                df_model, feature_cols = add_tree_lag_features(
                    df_model, forecast_horizon, calendar_cols=calendar_cols,
                )
                train_df, _ = add_tree_lag_features(
                    train_df, forecast_horizon, calendar_cols=calendar_cols,
                )
                df_model = df_model.dropna()
                train_df = train_df.dropna()

                # Guard: lag features may consume most rows on small datasets.
                # Use explicit train_range/test_range boundaries when available
                # so the check reflects the actual split rather than split_date.
                model_rows = df_model[df_model['dataset_name'] == dataset]
                _train_range = strategy_params.get('train_range')
                _test_range = strategy_params.get('test_range')
                if _train_range:
                    _tr_start, _tr_end = _train_range
                    _train_mask = pd.Series(True, index=model_rows.index)
                    if _tr_start:
                        _train_mask &= model_rows.index >= pd.to_datetime(_tr_start)
                    if _tr_end:
                        _train_mask &= model_rows.index <= pd.to_datetime(_tr_end)
                    train_rows = model_rows[_train_mask]
                else:
                    train_rows = model_rows[model_rows.index <= split_date]
                if _test_range:
                    _te_start, _te_end = _test_range
                    _test_mask = pd.Series(True, index=model_rows.index)
                    if _te_start:
                        _test_mask &= model_rows.index >= pd.to_datetime(_te_start)
                    if _te_end:
                        _test_mask &= model_rows.index <= pd.to_datetime(_te_end)
                    test_rows = model_rows[_test_mask]
                else:
                    test_rows = model_rows[model_rows.index > split_date]
                if train_rows.empty or test_rows.empty:
                    logger.warning(
                        "Skipping %s / %dd: not enough data after lag features "
                        "(train=%d, test=%d rows).",
                        dataset, forecast_horizon, len(train_rows), len(test_rows),
                    )
                    continue
            else:
                # Neural models: pass calendar columns as feature_columns
                feature_cols = list(calendar_cols)

            # 3. Train ──────────────────────────────────────────────────
            #
            # Rolling-backtest gate: when use_rolling_training=True the entire
            # train→predict loop is delegated to _run_rolling_backtest, which
            # retrains (or fine-tunes) the model on a sliding window at each
            # retrain interval.  Tree-based models are excluded (they rely on
            # pre-engineered lag features that are already aligned to the full
            # training history and don't support incremental fine-tuning).
            if (
                strategy_params.get('use_rolling_training', False)
                and model_type not in (
                    'xgboost', 'lightgbm',
                    # Hussain variants are article reproductions with
                    # schedule-mode prediction; rolling retraining would
                    # change their semantics and is excluded by design.
                    'hussain_lstm', 'hussain_transformer', 'hussain_hybrid',
                )
            ):
                look_back = strategy_params.get('look_back', 28)
                logger.info(
                    "Rolling backtest mode for %s / %dd (window=%d days, interval=%d days, mode=%s).",
                    dataset, forecast_horizon,
                    strategy_params.get('rolling_window_days', 180),
                    strategy_params.get('rolling_retrain_interval', 7),
                    strategy_params.get('rolling_retrain_mode', 'full'),
                )
                preds_array, actuals_array, predict_dates_idx = _run_rolling_backtest(
                    df_full=df_model,
                    dataset=dataset,
                    strategy=strategy,
                    strategy_params=strategy_params,
                    params_key=params_key,
                    feature_cols=feature_cols,
                    look_back=look_back,
                    split_date_str=split_date_str,
                    model_name_prefix=model_name_prefix,
                )
                if len(preds_array) == 0:
                    logger.warning(
                        "Skipping %s / %dd: rolling backtest returned no predictions.",
                        dataset, forecast_horizon,
                    )
                    continue

                model_name = f"{model_name_prefix}_{dataset}{model_suffix}"
                # Skip save_model — rolling models are ephemeral by design.

                # Inverse target transforms — metrics are in original kWh scale
                min_len = min(len(preds_array), len(actuals_array))
                p = preds_array[:min_len]
                a = actuals_array[:min_len]
                predict_dates_idx = predict_dates_idx[:min_len]
                p, a = apply_inverse_transforms(p, a, predict_dates_idx, transform_state)
                p = np.clip(p, 0.0, None)  # energy demand is non-negative
                plot_actuals_df = pd.DataFrame({'y': a}, index=predict_dates_idx)

                # Drop NaN pairs
                valid_mask = ~(np.isnan(a) | np.isnan(p))
                if not valid_mask.all():
                    n_nan = (~valid_mask).sum()
                    logger.warning(
                        "%s / %dd: dropping %d NaN rows after inverse transforms (%d remain).",
                        dataset, forecast_horizon, n_nan, valid_mask.sum(),
                    )
                    a, p = a[valid_mask], p[valid_mask]
                    predict_dates_idx = predict_dates_idx[valid_mask]
                    plot_actuals_df = pd.DataFrame({'y': a}, index=predict_dates_idx)
                if len(a) == 0:
                    logger.warning("Skipping %s / %dd: no valid data after NaN removal.", dataset, forecast_horizon)
                    continue

                mse_val = mean_squared_error(a, p)
                metrics = {
                    'MSE': mse_val,
                    'RMSE': math.sqrt(mse_val),
                    'MAE': mean_absolute_error(a, p),
                    'MAPE': mean_absolute_percentage_error(a, p),
                    'SMAPE': smape(p, a),
                    'MASE': mase(p, a, train_df_plot['y'].values),
                }
                logger.info("Metrics for %s (%d days): %s", dataset, forecast_horizon, metrics)
                if metrics['SMAPE'] >= 50.0:
                    logger.warning(
                        "sMAPE=%.1f%% ≥ 50%% for %s / %dd: model is not outperforming the "
                        "naive 'predict yesterday' baseline (trivial baseline). "
                        "Predictions may be uninformative.",
                        metrics['SMAPE'], dataset, forecast_horizon,
                    )
                horizon_metrics[forecast_horizon] = metrics
                results_summary.append(
                    _build_metrics_record(
                        experiment_name=model_name,
                        model_type=model_type,
                        dataset=dataset,
                        forecast_horizon=forecast_horizon,
                        split_date_str=split_date_str,
                        strategy_params=strategy_params,
                        metrics=metrics,
                    )
                )
                zoom_range = strategy_params.get('zoom_range')
                plot_path_predict = plot_test_vs_predict(
                    plot_actuals_df, p, dataset, model_name, forecast_horizon,
                    plots_dir=plots_dir,
                )
                continue  # skip the standard train→predict path below
            # ── end rolling backtest gate ────────────────────────────

            logger.info("Training model with split_date=%s...", split_date_str)
            trained_models_dict = strategy.train(
                df=df_model,
                feature_columns=feature_cols,
                target_column='y',
                dataset_names=[dataset],
                model_name_prefix=model_name_prefix,
                split_date=split_date_str,
            )
            trained_model_objects = trained_models_dict['train'][f"{model_name_prefix}_{dataset}"]['model']

            model_name = f"{model_name_prefix}_{dataset}{model_suffix}"
            save_model(model_name, trained_models_dict, models_dir)

            # 4. Predict ────────────────────────────────────────────────
            logger.info("Generating predictions...")

            if model_type in ('xgboost', 'lightgbm'):
                predict_df = test_df.copy()
                predict_df['dataset_name'] = dataset
                common_idx = predict_df.index.intersection(df_model.index)
                # Only copy feature columns not already present in predict_df
                # (calendar cols are already in test_df via add_calendar_features;
                # pulling them again from df_model would produce duplicate columns).
                missing_feat_cols = [c for c in feature_cols if c not in predict_df.columns]
                lag_cols_df = pd.DataFrame(df_model.loc[common_idx, missing_feat_cols])
                predict_df = pd.concat([predict_df, lag_cols_df], axis=1).dropna()
                predict_mode = None
            else:
                predict_df = test_df.copy()
                predict_df['dataset_name'] = dataset
                predict_mode = None  # backtest / validate

            predict_dict = strategy.predict(
                df=predict_df,
                feature_columns=feature_cols,
                target_column='y',
                model_objects=trained_model_objects,
                context_date=None,
                dataset_names=[dataset],
                submode=predict_mode,
            )
            if 'values' not in predict_dict.get('predict', {}):
                logger.warning(
                    "Skipping %s / %dd: predict returned no values.",
                    dataset, forecast_horizon,
                )
                continue
            predictions = predict_dict['predict']['values']
            predict_dates = predict_dict['predict'].get('dates')
            predict_actuals = predict_dict['predict'].get('actuals')

            # 5. Metrics ────────────────────────────────────────────────
            preds_array = np.array(predictions)
            # Tree models predict from engineered features — no warm-up window.
            # Neural models store look_back in the model-objects dict.
            if model_type in ('xgboost', 'lightgbm'):
                look_back = 0
            else:
                look_back = trained_model_objects.get('look_back', 30)

            if predict_dates:
                predict_dates_idx = pd.to_datetime(predict_dates)
                if predict_actuals is not None:
                    actuals_array = np.array(predict_actuals)
                else:
                    available_mask = predict_dates_idx.isin(df.index)
                    common_dates = predict_dates_idx[available_mask]
                    actuals_array = df.loc[common_dates, 'y'].values
                    preds_array = preds_array[available_mask]
                    predict_dates_idx = common_dates
                min_len = min(len(preds_array), len(actuals_array))
                a, p = actuals_array[:min_len], preds_array[:min_len]
                plot_actuals_df = pd.DataFrame({'y': a}, index=predict_dates_idx[:min_len])
            else:
                # Fallback: _predict_backtest returns predictions starting at
                # test[look_back:], so align actuals and dates accordingly.
                actuals_array = test_df['y'].values[look_back:]
                min_len = min(len(preds_array), len(actuals_array))
                a, p = actuals_array[:min_len], preds_array[:min_len]
                predict_dates_idx = test_df.index[look_back:look_back + min_len]
                plot_actuals_df = pd.DataFrame({'y': a}, index=predict_dates_idx)

            # Inverse target transforms — metrics are in original kWh scale
            p, a = apply_inverse_transforms(p, a, predict_dates_idx[:min_len], transform_state)
            p = np.clip(p, 0.0, None)  # energy demand is non-negative
            plot_actuals_df = pd.DataFrame({'y': a}, index=predict_dates_idx[:min_len])

            # Drop NaN pairs (inverse transforms may produce NaN on boundary dates)
            valid_mask = ~(np.isnan(a) | np.isnan(p))
            if not valid_mask.all():
                n_nan = (~valid_mask).sum()
                logger.warning(
                    "%s / %dd: dropping %d NaN rows after inverse transforms (%d remain).",
                    dataset, forecast_horizon, n_nan, valid_mask.sum(),
                )
                a, p = a[valid_mask], p[valid_mask]
                predict_dates_idx = predict_dates_idx[valid_mask]
                plot_actuals_df = pd.DataFrame({'y': a}, index=predict_dates_idx)
            if len(a) == 0:
                logger.warning("Skipping %s / %dd: no valid data after NaN removal.", dataset, forecast_horizon)
                continue

            mse_val = mean_squared_error(a, p)
            metrics = {
                'MSE': mse_val,
                # RMSE included because Hussain et al. (2025) appear to report
                # RMSE labelled as "MSE" — their values are only slightly above
                # their MAE, consistent with √MSE but not raw MSE.
                'RMSE': math.sqrt(mse_val),
                'MAE': mean_absolute_error(a, p),
                'MAPE': mean_absolute_percentage_error(a, p),
                'SMAPE': smape(p, a),
                'MASE': mase(p, a, train_df_plot['y'].values),
            }
            logger.info("Metrics for %s (%d days): %s", dataset, forecast_horizon, metrics)
            if metrics['SMAPE'] >= 50.0:
                logger.warning(
                    "sMAPE=%.1f%% ≥ 50%% for %s / %dd: model is not outperforming the "
                    "naive 'predict yesterday' baseline (trivial baseline). "
                    "Predictions may be uninformative.",
                    metrics['SMAPE'], dataset, forecast_horizon,
                )
            logger.info(
                "Evaluation window: %s to %s (%d points, look_back=%d warm-up excluded)",
                predict_dates_idx[0].strftime('%Y-%m-%d'),
                predict_dates_idx[-1].strftime('%Y-%m-%d'),
                len(a), look_back,
            )
            horizon_metrics[forecast_horizon] = metrics

            results_summary.append(
                _build_metrics_record(
                    experiment_name=model_name,
                    model_type=model_type,
                    dataset=dataset,
                    forecast_horizon=forecast_horizon,
                    split_date_str=split_date_str,
                    strategy_params=strategy_params,
                    metrics=metrics,
                )
            )

            # 6. Plots ─────────────────────────────────────────────────
            zoom_range = strategy_params.get('zoom_range')
            warm_up = look_back if model_type not in ('xgboost', 'lightgbm') else 0
            plot_path_split = plot_train_test_split(
                train_df_plot, test_df_plot, dataset, forecast_horizon,
                zoom_range=zoom_range, model_name=model_name,
                warm_up_days=warm_up, plots_dir=plots_dir,
            )
            plot_path_predict = plot_test_vs_predict(
                plot_actuals_df, p, dataset, model_name, forecast_horizon,
                plots_dir=plots_dir,
            )

            # 7. MLflow (optional) ─────────────────────────────────────
            if mlflow_tracking_uri and _MLFLOW_AVAILABLE and False:
                try:
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                    experiment_name = model_name
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment is None:
                        experiment_id = mlflow.create_experiment(experiment_name)
                    else:
                        experiment_id = experiment.experiment_id

                    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_name}_{forecast_horizon}d"):
                        for k, v in strategy_params.items():
                            if isinstance(v, (int, float, str, bool)):
                                mlflow.log_param(k, v)
                        mlflow.log_param('model_type', model_type)
                        mlflow.log_param('split_date', split_date_str)
                        mlflow.log_param('forecast_horizon', forecast_horizon)
                        mlflow.log_metrics(metrics)

                        train_output = trained_models_dict['train'].get(f"{model_name_prefix}_{dataset}", {})
                        model_obj = train_output.get('model')
                        if isinstance(model_obj, dict) and 'keras_model' in model_obj:
                            mlflow.keras.log_model(
                                model_obj['keras_model'],
                                artifact_path=f"{model_type}_model",
                                registered_model_name=model_name,
                            )
                            with tempfile.TemporaryDirectory() as tmpdir:
                                if model_obj.get('scaler') is not None:
                                    scaler_p = os.path.join(tmpdir, 'scaler.pkl')
                                    joblib.dump(model_obj['scaler'], scaler_p)
                                    mlflow.log_artifact(scaler_p, artifact_path=f"{model_type}_model")
                                meta = {
                                    'look_back': model_obj.get('look_back', 30),
                                    'params': strategy_params,
                                }
                                meta_p = os.path.join(tmpdir, 'meta.pkl')
                                joblib.dump(meta, meta_p)
                                mlflow.log_artifact(meta_p, artifact_path=f"{model_type}_model")
                        elif model_obj is not None:
                            try:
                                if model_type == 'lightgbm':
                                    mlflow.lightgbm.log_model(
                                        model_obj, artifact_path=f"{model_type}_model",
                                        registered_model_name=model_name,
                                    )
                                elif model_type == 'xgboost':
                                    mlflow.xgboost.log_model(
                                        model_obj, artifact_path=f"{model_type}_model",
                                        registered_model_name=model_name,
                                    )
                            except Exception as model_log_err:
                                logger.warning("Could not log model to MLflow: %s", model_log_err)

                        if os.path.exists(plot_path_predict):
                            mlflow.log_artifact(plot_path_predict, artifact_path="plots")
                        if os.path.exists(plot_path_split):
                            mlflow.log_artifact(plot_path_split, artifact_path="plots")

                    logger.info("MLflow run logged for %s", model_name)
                except Exception as mlflow_err:
                    logger.warning("MLflow logging failed (non-blocking): %s", mlflow_err)

    summary_df = pd.DataFrame(results_summary)
    metric_cols = ['experiment_name', 'dataset', 'forecast_horizon', 'MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE']
    display_cols = [c for c in metric_cols if c in summary_df.columns]
    logger.info("\nFinal Benchmark Summary:\n%s", summary_df[display_cols].to_string())
    if results_summary:
        _upsert_metrics_csv(results_summary)
    return horizon_metrics
