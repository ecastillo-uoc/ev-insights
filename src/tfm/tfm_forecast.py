"""
EV charging-station demand forecasting — experiment orchestrator.

This module is the **top-level entry point** for running forecasting
experiments.  It defines:

* **Dataset configurations** — train/test splits and zoom ranges for each
  dataset, both our own and the Hussain et al. (2025) article variants.
* **``execute_*`` functions** — one per model type, each wiring the
  appropriate ``strategy_params`` and calling
  :func:`~src.forecast.pipeline.run_forecast_pipeline`.
* **``optimize_lstm``** — Optuna-based hyper-parameter search.

All computational logic lives elsewhere:

+-----------------------------------------+-------------------------------------------+
| Responsibility                          | Module                                    |
+=========================================+===========================================+
| Feature engineering (calendar, rolling, | ``src.forecast.feature_engineering``      |
| lags, target transforms)                |                                           |
+-----------------------------------------+-------------------------------------------+
| Pipeline orchestration (train → predict | ``src.forecast.pipeline``                 |
| → metrics → plots → MLflow)             |                                           |
+-----------------------------------------+-------------------------------------------+
| Model architectures                     | ``src.forecast.strategies.*``             |
+-----------------------------------------+-------------------------------------------+

Three boolean flags in ``strategy_params`` control optional pre-processing
applied uniformly to **all** strategies:

* ``use_log_transform``  — ``log1p`` / ``expm1`` round-trip on target
* ``use_differencing``   — first-order diff / per-date reconstruction
* ``use_calendar_features`` — day-of-week, month, sin/cos, business day

All default to ``False`` so existing behaviour is unchanged.  They can be
combined freely, yielding up to 8 experiment configurations.
"""
import json
import os
import sys
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add src to python path if not present
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.gpu_config import configure_gpu
configure_gpu()

from src.utils.console import Colors
from src.forecast.pipeline import run_forecast_pipeline
from src.data.data_fetcher import fetch_daily_energy_for_forecast

# Optimization
import optuna
from src.forecast.optimization import objective_lstm, objective_transformer

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Minimum look-back window for all non-Hussain neural models.
# For horizons shorter than this, look_back is clamped to MIN_LOOK_BACK.
# For horizons equal to or longer than this, look_back = forecast_horizon.
MIN_LOOK_BACK = 14
# Fixed context window for our non-Hussain neural models.
# Decoupled from forecast_horizon: 28 days captures two full weekly cycles,
# covers significant ACF lags (7, 14, 28 days) and avoids anchoring predictions
# to a stale demand regime when long horizons are used.
LOOK_BACK_DAYS = 28


def get_dataset_config(dataset_name, hussain=False):
    """Return (split_date_str, ranges_dict) for a given dataset.

    ranges_dict may contain 'train_range', 'test_range', 'zoom_range'.
    When a key is absent, run_forecast_pipeline uses the split_date fallback.
    """
    if hussain:
        # Hussain et al. article splits (COVID data intentionally included)
        _configs = {
            'ACN_JPL': {
                'split_date': '2020-08-05',
                'train_range': ('2018-09-01', '2020-08-05'),
                'test_range': ('2020-11-17', '2021-03-31'),
                'zoom_range': ('2021-01-01', '2021-03-31'),
            },
            'ACN_Caltech': {
                'split_date': '2020-08-05',
                'train_range': ('2018-09-01', '2020-08-05'),
                'test_range': ('2020-11-17', '2021-03-31'),
                'zoom_range': ('2021-01-01', '2021-03-31'),
            },
            'Dundee': {
                'split_date': '2023-09-01',
                'zoom_range': ('2023-09-01', '2024-06-01'),
            },
        }
    else:
        _configs_pre_covid = {
            'ACN_JPL': {
                'split_date': '2019-10-01',
                'train_range': ('2019-01-01', '2019-09-30'),
                'test_range': ('2019-10-01', '2020-03-01'),
                'zoom_range': ('2019-10-01', '2020-03-01'),
            },
            'ACN_Caltech': {
                'split_date': '2019-07-01',
                'train_range': ('2019-01-01', '2019-06-30'),
                'test_range': ('2019-07-01', '2019-12-15'),
                'zoom_range': ('2019-07-01', '2019-12-15'),
            },
            'Dundee': {
                'split_date': '2023-09-01',
                'zoom_range': ('2023-09-01', '2024-06-01'),
            },
        }

        _configs = {
            'ACN_JPL': {
                'split_date': '2020-08-05',
                'train_range': ('2018-09-01', '2020-08-05'),
                'test_range': ('2020-11-17', '2021-03-31'),
                'zoom_range': ('2021-01-01', '2021-03-31'),
            },
            'ACN_Caltech': {
                'split_date': '2020-08-05',
                'train_range': ('2018-09-01', '2020-08-05'),
                'test_range': ('2020-11-17', '2021-03-31'),
                'zoom_range': ('2021-01-01', '2021-03-31'),
            },
            'ACN_Office001': {
                'split_date': '2020-07-31',
                'train_range': ('2019-03-26', '2020-07-31'),
                'test_range': ('2020-08-21', '2021-09-14'),
                'zoom_range': ('2020-08-21', '2021-03-31'),
            },
            'Dundee': {
                'split_date': '2023-09-01',
                'zoom_range': ('2023-09-01', '2024-06-01'),
            },
            'BeLib': {
                'split_date': '2017-04-24',
                'train_range': ('2017-04-01', '2017-04-24'),
                'test_range': ('2017-04-25', '2017-04-30'),
                'zoom_range': ('2017-04-25', '2017-04-30'),
            },
            'AMB_Barcelona': {
                'split_date': '2019-10-19',
                'train_range': ('2018-12-31', '2019-10-19'),
                'test_range': ('2019-10-20', '2019-12-31'),
                'zoom_range': ('2019-10-20', '2019-12-31'),
            },
        }

    if dataset_name not in _configs:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Known: {list(_configs.keys())}")

    cfg = _configs[dataset_name]
    split_date = cfg['split_date']
    ranges = {k: v for k, v in cfg.items() if k != 'split_date'}
    return split_date, ranges


def execute_lstm(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    """Run LSTM forecasts for each dataset × forecast horizon.

    Parameters
    ----------
    datasets : list[str]
        Dataset identifiers (e.g. ``['ACN_Caltech', 'ACN_JPL']``).
    li_forecast_horizons : list[int]
        Forecast horizons in days (e.g. ``[1, 7, 30, 120]``).
    mlflow_tracking_uri : str or None
        MLflow server URI.  ``None`` disables tracking.
    """
    for ds in datasets:
        split_date_str, ranges = get_dataset_config(ds)
        for forecast_horizon in li_forecast_horizons:
            look_back = LOOK_BACK_DAYS
            strategy_params_lstm = {
                'epochs': 100,
                'batch_size': 32,
                'look_back': look_back,
                'li_forecast_horizons': [forecast_horizon],
                'learning_rate': 0.001,
                'dropout_rate': 0.2,
                'activation': 'relu',
                'use_log_transform': True,
                'use_differencing': False,
                'use_calendar_features': False,
                # Rolling-training options (disabled by default; set
                # use_rolling_training=True to activate backtest mode).
                'use_rolling_training':     False,
                'rolling_window_days':      180,
                'rolling_retrain_interval': 7,
                'rolling_retrain_mode':     'full',
                'rolling_finetune_epochs':  10,
            }
            params = {**strategy_params_lstm, **ranges}
            run_forecast_pipeline([ds], params, split_date_str, mlflow_tracking_uri=mlflow_tracking_uri)

def execute_transformer(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    """Run Transformer forecasts for each dataset × forecast horizon.

    Parameters
    ----------
    datasets : list[str]
        Dataset identifiers.
    li_forecast_horizons : list[int]
        Forecast horizons in days.
    mlflow_tracking_uri : str or None
        MLflow server URI.  ``None`` disables tracking.
    """
    for ds in datasets:
        split_date_str, ranges = get_dataset_config(ds)
        for forecast_horizon in li_forecast_horizons:
            look_back = LOOK_BACK_DAYS
            strategy_params_transformer = {
                'epochs': 100,
                'batch_size': 32,
                'look_back': look_back,
                'li_forecast_horizons': [forecast_horizon],
                'learning_rate': 0.001,
                'head_size': 128,
                'num_heads': 4,
                'ff_dim': 4,
                'num_transformer_blocks': 2,
                'dropout': 0.1,
                'mlp_dropout': 0.1,
                'use_log_transform': True,
                'use_differencing': False,
                'use_calendar_features': False,
            }
            params = {**strategy_params_transformer, **ranges}
            run_forecast_pipeline([ds], params, split_date_str, model_type="transformer", mlflow_tracking_uri=mlflow_tracking_uri)

def execute_lightgbm(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    """Run LightGBM forecasts for each dataset × forecast horizon.

    Parameters
    ----------
    datasets : list[str]
        Dataset identifiers.
    li_forecast_horizons : list[int]
        Forecast horizons in days.
    mlflow_tracking_uri : str or None
        MLflow server URI.  ``None`` disables tracking.
    """
    strategy_params_lgbm = {
        'li_forecast_horizons': li_forecast_horizons,
        'num_leaves': 31,
        'learning_rate': 0.02,
        'max_depth': 8,
        'early_stopping_rounds': 200,
        'use_log_transform': True,
        'use_differencing': False,
        'use_calendar_features': False,
    }
    for ds in datasets:
        split_date_str, ranges = get_dataset_config(ds)
        params = {**strategy_params_lgbm, **ranges}
        run_forecast_pipeline([ds], params, split_date_str, model_type="lightgbm", mlflow_tracking_uri=mlflow_tracking_uri)

def execute_xgboost(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    """Run XGBoost forecasts for each dataset × forecast horizon.

    Parameters
    ----------
    datasets : list[str]
        Dataset identifiers.
    li_forecast_horizons : list[int]
        Forecast horizons in days.
    mlflow_tracking_uri : str or None
        MLflow server URI.  ``None`` disables tracking.
    """
    strategy_params_xgb = {
        'li_forecast_horizons': li_forecast_horizons,
        'random_state': 16,
        'test_size': 0.20,
        'use_log_transform': True,
        'use_differencing': False,
        'use_calendar_features': False,
    }
    for ds in datasets:
        split_date_str, ranges = get_dataset_config(ds)
        params = {**strategy_params_xgb, **ranges}
        run_forecast_pipeline([ds], params, split_date_str, model_type="xgboost", mlflow_tracking_uri=mlflow_tracking_uri)


def execute_hybrid(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    """Run Hybrid LSTM-Transformer forecasts for each dataset × forecast horizon.

    Parameters
    ----------
    datasets : list[str]
        Dataset identifiers.
    li_forecast_horizons : list[int]
        Forecast horizons in days.
    mlflow_tracking_uri : str or None
        MLflow server URI.  ``None`` disables tracking.
    """
    for ds in datasets:
        split_date_str, ranges = get_dataset_config(ds)
        for forecast_horizon in li_forecast_horizons:
            look_back = LOOK_BACK_DAYS
            strategy_params_hybrid = {
                'epochs': 100,
                'batch_size': 32,
                'look_back': look_back,
                'li_forecast_horizons': [forecast_horizon],
                'learning_rate': 0.001,
                'd_model': 128,
                'num_heads': 4,
                'dropout': 0.1,
                'use_log_transform': True,
                'use_differencing': False,
                'use_calendar_features': False,
                # Rolling-training options (disabled by default; set
                # use_rolling_training=True to activate backtest mode).
                'use_rolling_training':     False,
                'rolling_window_days':      180,
                'rolling_retrain_interval': 7,
                'rolling_retrain_mode':     'full',
                'rolling_finetune_epochs':  10,
            }
            params = {**strategy_params_hybrid, **ranges}
            run_forecast_pipeline([ds], params, split_date_str, model_type="hybrid", mlflow_tracking_uri=mlflow_tracking_uri)


# =============================================================================
# Hussain et al. (2025) article-variant models
# --------------------------------------------------------------------------
# These functions reproduce the architectures and hyperparameters described in:
#   Hussain, A. et al. "Charging stations demand forecasting using LSTM based
#   hybrid transformer model." Sci Rep 15, 13555 (2025).
#
# Key differences from our default models:
#   * look_back = max(MIN_LOOK_BACK, forecast_horizon) — floor at 14 days; article uses h=h
#   * dropout = 0.2   (vs. 0.1 for our Transformer/Hybrid)
#   * ReduceLROnPlateau + EarlyStopping callbacks (article text, Section IV)
#   * Transformer uses a single Dense(ReLU) → MHA → GAP → Dense(1) arch
#     (no residual, no LayerNorm, no feed-forward block, no MLP head)
#   * Hybrid uses HussainHybridModelStrategy which removes residual
#     connections and adds positional encoding (matching Fig. 4)
#   * Prediction uses schedule (multi-step recursive) mode matching the
#     article figures (Figs 9–26 show exactly N predicted days)
#   * COVID data is intentionally INCLUDED (exclude_covid=False) to match
#     the article methodology — visible in their Fig. 8 train/test split
#
# IMPORTANT — "MSE" column in the article:
#   The paper's reported "MSE" values are numerically very close to their MAE
#   (e.g. MAE=82.8, "MSE"=90.5 for JPL 30d LSTM).  This is inconsistent with
#   true MSE (should be >>MAE²/N).  We believe the article reports RMSE (√MSE)
#   mislabelled as MSE.  Our RMSE column is the correct comparison target.
# =============================================================================

def execute_hussain_lstm(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    """LSTM with Hussain et al. hyperparams: look_back = max(MIN_LOOK_BACK, forecast_horizon), dropout=0.2."""
    for ds in datasets:
        split_date_str, ranges = get_dataset_config(ds, hussain=True)
        for forecast_horizon in li_forecast_horizons:
            strategy_params = {
                'epochs': 100,
                'batch_size': 32,
                'look_back': max(MIN_LOOK_BACK, forecast_horizon),  # Article uses h=h, but floor at 14d
                'li_forecast_horizons': [forecast_horizon],
                'learning_rate': 0.001,
                'dropout_rate': 0.2,              # Article Table 1
                'activation': 'relu',
                'use_lr_scheduler': True,          # Article: ReduceLROnPlateau
                'use_early_stopping': True,        # Article: EarlyStopping
                'use_log_transform': True,
                'use_differencing': False,
                'use_calendar_features': False,
            }
            params = {**strategy_params, **ranges}
            run_forecast_pipeline([ds], params, split_date_str,
                                  model_type="hussain_lstm", mlflow_tracking_uri=mlflow_tracking_uri)


def execute_hussain_transformer(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    """Simplified Transformer from Hussain et al.: Dense→MHA→GAP→Dense(1)."""
    for ds in datasets:
        split_date_str, ranges = get_dataset_config(ds, hussain=True)
        for forecast_horizon in li_forecast_horizons:
            strategy_params = {
                'epochs': 100,
                'batch_size': 32,
                'look_back': max(MIN_LOOK_BACK, forecast_horizon),  # Article uses h=h, but floor at 14d
                'li_forecast_horizons': [forecast_horizon],
                'learning_rate': 0.001,
                'encoding_dim': 64,
                'num_heads': 4,
                'key_dim': 64,
                'dropout': 0.2,                   # Article Table 1
                'use_lr_scheduler': True,          # Article: ReduceLROnPlateau
                'use_early_stopping': True,        # Article: EarlyStopping
                'use_log_transform': True,
                'use_differencing': False,
                'use_calendar_features': False,
            }
            params = {**strategy_params, **ranges}
            run_forecast_pipeline([ds], params, split_date_str,
                                  model_type="hussain_transformer", mlflow_tracking_uri=mlflow_tracking_uri)


def execute_hussain_hybrid(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    """Hybrid LSTM-Transformer with Hussain et al. hyperparams: look_back = max(MIN_LOOK_BACK, forecast_horizon), dropout=0.2."""
    for ds in datasets:
        split_date_str, ranges = get_dataset_config(ds, hussain=True)
        for forecast_horizon in li_forecast_horizons:
            strategy_params = {
                'epochs': 100,
                'batch_size': 32,
                'look_back': max(MIN_LOOK_BACK, forecast_horizon),  # Article uses h=h, but floor at 14d
                'li_forecast_horizons': [forecast_horizon],
                'learning_rate': 0.001,
                'd_model': 128,
                'num_heads': 4,
                'dropout': 0.2,                   # Article Table 1
                'use_lr_scheduler': True,          # Article: ReduceLROnPlateau
                'use_early_stopping': True,        # Article: EarlyStopping
                'use_log_transform': True,
                'use_differencing': False,
                'use_calendar_features': False,
            }
            params = {**strategy_params, **ranges}
            run_forecast_pipeline([ds], params, split_date_str,
                                  model_type="hussain_hybrid", mlflow_tracking_uri=mlflow_tracking_uri)


def optimize_lstm(datasets, n_trials, mlflow_tracking_uri=None):
    """Run Optuna LSTM hyperparameter search for a single dataset.

    Uses :func:`~src.forecast.optimization.objective_lstm` as the Optuna
    objective.  After the study completes, retrains a final model with the
    best hyperparameters and saves it to ``output_models/``.

    Parameters
    ----------
    datasets : list[str]
        Dataset identifiers.  Only the **first** element is used.
    n_trials : int
        Number of Optuna trials.
    mlflow_tracking_uri : str or None
        MLflow server URI.  ``None`` disables tracking.

    .. warning::
       ``split_date_str`` and ``li_forecast_horizons`` are currently read as
       free variables from the ``if __name__ == '__main__'`` block.  Calling
       this function from outside that block will raise ``NameError``.
    """
    # Extract dataset name from list
    dataset_name = datasets[0] if datasets else None
    if not dataset_name:
        logging.error("No dataset specified. Exiting.")
        sys.exit(1)
    
    logging.info(f"Starting {Colors.GREEN}LSTM{Colors.NORMAL} hyperparameter optimization for {Colors.GREEN}{dataset_name}{Colors.NORMAL}")
    
    # =========================================================================
    # 1. Fetch and prepare data
    # =========================================================================
    logging.info("Fetching data...")
    df_full = fetch_daily_energy_for_forecast(dataset_name)
    
    if df_full.empty:
        logging.error(f"No data found for {dataset_name}. Exiting.")
        sys.exit(1)
    
    logging.info(f"Data shape: {df_full.shape}, date range: {df_full.index.min()} to {df_full.index.max()}")
    
    # =========================================================================
    # 2. Prepare training data and run Optuna study
    # =========================================================================
    split_date = pd.to_datetime(split_date_str)
    train_df = df_full[df_full.index <= split_date].copy()
    train_df['y'] = train_df[train_df.columns[0]]  # Ensure 'y' column exists
    target_column = 'y'
    
    logging.info(f"Training data shape: {train_df.shape}, date range: {train_df.index.min()} to {train_df.index.max()}")
    
    # =========================================================================
    # 3. Run Optuna study
    # =========================================================================
    logging.info(f"Starting Optuna optimization with {n_trials} trials...")
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.SuccessiveHalvingPruner(reduction_factor=3)
    )
    
    # Use lambda wrapper to pass data to objective function
    study.optimize(
        lambda trial: objective_lstm(trial, train_df, target_column), 
        n_trials=n_trials, 
        show_progress_bar=True
    )
    
    # =========================================================================
    # 4. Retrieve and display best results
    # =========================================================================
    best_trial = study.best_trial
    best_params = best_trial.user_attrs["lstm_params"]
    best_smape = best_trial.value
    
    logging.info("\n" + "="*80)
    logging.info("OPTIMIZATION COMPLETE - BEST PARAMETERS")
    logging.info("="*80)
    logging.info(f"Best sMAPE: {best_smape:.4f}")
    logging.info(f"Best Trial Number: {best_trial.number}")
    logging.info("Best Hyperparameters:")
    for param, value in best_params.items():
        logging.info(f"  {param}: {value}")
    logging.info("="*80 + "\n")
    
    # =========================================================================
    # 5. Train final model with best parameters
    # =========================================================================
    logging.info("Training final model with best parameters...")
    
    strategy_params_lstm_best = {
        **best_params,
        'li_forecast_horizons': li_forecast_horizons,
    }
    
    strategy_jpl = {
        'train_range': ('2019-01-01', '2019-09-30'), 
        'test_range': ('2019-10-01', '2020-03-01'), 
        'zoom_range': ('2019-10-01', '2020-03-01') 
    }

    strategy_params_lstm_best_jpl = {**strategy_params_lstm_best, **strategy_jpl}
    
    run_forecast_pipeline([dataset_name], strategy_params_lstm_best_jpl, split_date_str, model_type="lstm", model_suffix="_optuna",
                           mlflow_tracking_uri=mlflow_tracking_uri)
    
    logging.info("\n" + "="*80)
    logging.info("LSTM OPTIMIZATION AND TRAINING COMPLETE")
    logging.info("="*80)
    logging.info(f"Results saved to:")
    logging.info(f"  - Models: output_models/")
    logging.info(f"  - Plots: output_plots/")
    logging.info(f"  - Metrics: forecast_metrics_summary.csv")
    logging.info("="*80)


# =============================================================================
# Full-set batch execution framework
# =============================================================================
#
# Usage:
#   from src.tfm.tfm_forecast import run_all_cases, ALL_DATASETS, ALL_FE_TAGS
#   results = run_all_cases()          # runs all 72 cases
#   results = run_all_cases(           # selective run
#       datasets=['Dundee'],
#       models=['lstm', 'lightgbm'],
#       fe_tags=['log'],
#   )
#
# Each case produces artifacts in:
#   tfm/doc/vf/chapters/results/{dataset}_{model}_{fe_tag}/
#     metadata.json   — config + per-horizon metrics + health flags
#     case.tex        — LaTeX subsection fragment
#     {dataset}_actual_vs_predict_{model}_{h}d_{dataset}_{h}days.png  (×4)
#     {dataset}_train_test_split_{model}_{h}d_{dataset}_{h}days.png   (×4)
# =============================================================================

# ── Constants ──────────────────────────────────────────────────────────────

# Directory under which per-case artifact folders are created
_TFM_CHAPTERS_DIR = (
    Path(__file__).resolve().parents[3]
    / 'tfm' / 'doc' / 'vf' / 'chapters'
)

ALL_DATASETS: List[str] = [
    'ACN_Caltech', 
    'Dundee',
    'ACN_JPL']
ALL_MODELS: List[str] = [
    'lightgbm',
    'xgboost',
    'lstm', 
    'transformer', 
    'hybrid',
    'hussain_lstm', 
    'hussain_transformer', 
    'hussain_hybrid',
]
ALL_FE_TAGS: List[str] = [
    'revin',
    'log_revin',
    'log_revin_cal',
    #'',
    'diff', 
    'diff_cal',
    #'log', 
    #'log_cal', 
    'log_diff', 
    'log_diff_cal',
    #'log_rolling',
    #'log_diff_cal_rolling'
    ]
ALL_HORIZONS: List[int] = [1, 7, 30, 120]

# Models that use hussain=True splits and the Hussain architecture variants
_HUSSAIN_MODELS: frozenset = frozenset({'hussain_lstm', 'hussain_transformer', 'hussain_hybrid'})
# Models that receive all horizons in a single pipeline call
_TREE_MODELS: frozenset = frozenset({'lightgbm', 'xgboost'})
# FE tags that activate rolling retraining — only meaningful for standard neural models.
# Hussain variants (article reproductions) and tree models are excluded at run time.
_ROLLING_FE_TAGS: frozenset = frozenset({'log_rolling', 'log_diff_cal_rolling'})
# Datasets where rolling retraining is expected to help due to a known demand
# regime shift between the training and test windows:
#   ACN_Caltech / ACN_JPL — COVID gap (2020-08-05 → 2020-11-17) + demand
#                            regime change during lockdown.
#   Dundee                — ~40 % demand drop after 2023-09-01 split.
# Other datasets (ACN_Office001, BeLib, AMB_Barcelona) show no evidence of
# systematic covariate shift and are excluded to save compute.
_ROLLING_DATASETS: frozenset = frozenset({'ACN_Caltech', 'ACN_JPL', 'Dundee'})

# FE configurations — three static baselines plus two rolling-retrain variants.
#
# Rolling variants activate the sliding-window backtest path in
# run_forecast_pipeline (use_rolling_training=True).  They are only run for
# standard neural models (lstm, transformer, hybrid); Hussain article-variant
# models and tree-based models are excluded at the run_all_cases level.
FE_VARIANTS: Dict[str, dict] = {
    '': {
        'use_log_transform': False,
        'use_differencing': False,
        'use_calendar_features': False,
    },
    'log': {
        'use_log_transform': True,
        'use_differencing': False,
        'use_calendar_features': False,
    },
    'log_cal': {
        'use_log_transform': True,
        'use_differencing': False,
        'use_calendar_features': True,
    },
    # ── Weekly differencing variants ─────────────────────────────────────
    # 7-day seasonal differencing removes weekly periodicity without
    # predicting yesterday's level.  Paired with log for stationarity.
    'log_diff': {
        'use_log_transform': True,
        'use_differencing': True,
        'use_calendar_features': False,
    },
    'log_diff_cal': {
        'use_log_transform': True,
        'use_differencing': True,
        'use_calendar_features': True,
    },
    'diff': {
        'use_log_transform': False,
        'use_differencing': True,
        'use_calendar_features': False,
    },
    'diff_cal': {
        'use_log_transform': False,
        'use_differencing': True,
        'use_calendar_features': True,
    },
    # ── Rolling variants ────────────────────────────────────────────────
    # log_rolling: minimal ablation — isolates rolling-retrain contribution
    # against the log baseline.  Direct comparison for covariate-shift
    # mitigation without confounding FE changes.
    'log_rolling': {
        'use_log_transform': True,
        'use_differencing': False,
        'use_calendar_features': False,
        'use_rolling_training':     True,
        'rolling_window_days':      180,   # ~6 months: two quarterly cycles
        'rolling_retrain_interval': 7,     # retrain every week
        'rolling_retrain_mode':     'full',
        'rolling_finetune_epochs':  10,
    },
    # log_diff_cal_rolling: maximal combination — differencing stabilises
    # each rolling window (less level-drift sensitivity at boundary);
    # calendar compensates for the level information lost by differencing;
    # rolling re-anchors the reconstruction baseline each retrain interval.
    'log_diff_cal_rolling': {
        'use_log_transform': True,
        'use_differencing': True,
        'use_calendar_features': True,
        'use_rolling_training':     True,
        'rolling_window_days':      180,
        'rolling_retrain_interval': 7,
        'rolling_retrain_mode':     'full',
        'rolling_finetune_epochs':  10,
    },

    # ── Per-window normalisation (RevIN-style) variants ──────────────────
    # Standardises each look-back window by its local mean and std.
    # No global scaler bias; compatible with log and calendar features.
    'revin': {
        'use_log_transform': False,
        'use_differencing': False,
        'use_calendar_features': False,
        'use_window_norm': True,
        'window_norm_days': 28,
    },
    'log_revin': {
        'use_log_transform': True,
        'use_differencing': False,
        'use_calendar_features': False,
        'use_window_norm': True,
        'window_norm_days': 28,
    },
    'log_revin_cal': {
        'use_log_transform': True,
        'use_differencing': False,
        'use_calendar_features': True,
        'use_window_norm': True,
        'window_norm_days': 28,
    },
}


# ── CaseConfig dataclass ───────────────────────────────────────────────────

@dataclass
class CaseConfig:
    """Configuration for one (dataset × model × feature_engineering) experiment case."""

    dataset: str
    model_type: str
    fe_tag: str
    fe_params: dict
    li_forecast_horizons: List[int]
    output_base_dir: Path
    hussain: bool = field(init=False)

    def __post_init__(self):
        self.hussain = self.model_type in _HUSSAIN_MODELS

    @property
    def case_id(self) -> str:
        return f"{self.dataset}_{self.model_type}_{self.fe_tag}"

    @property
    def case_dir(self) -> Path:
        return self.output_base_dir / self.case_id


# ── Strategy-params builder ────────────────────────────────────────────────

def _build_strategy_params_for_case(
    model_type: str,
    fe_params: dict,
    horizons: List[int],
) -> dict:
    """Return base strategy_params for a case, merging FE flags into model defaults.

    The returned dict does **not** include ``look_back`` for neural models —
    that is set per-horizon inside :func:`run_single_case`.
    """
    is_hussain = model_type in _HUSSAIN_MODELS
    base: dict = {
        'li_forecast_horizons': list(horizons),
        **fe_params,
    }

    if model_type in ('lstm', 'hussain_lstm'):
        base.update({
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.2,
            'activation': 'relu',
        })
        if is_hussain:
            base.update({'use_lr_scheduler': True, 'use_early_stopping': True})

    elif model_type == 'transformer':
        base.update({
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'head_size': 128,
            'num_heads': 4,
            'ff_dim': 4,
            'num_transformer_blocks': 2,
            'dropout': 0.1,
            'mlp_dropout': 0.1,
        })

    elif model_type == 'hussain_transformer':
        base.update({
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'encoding_dim': 64,
            'num_heads': 4,
            'key_dim': 64,
            'dropout': 0.2,
            'use_lr_scheduler': True,
            'use_early_stopping': True,
        })

    elif model_type == 'hybrid':
        base.update({
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'd_model': 128,
            'num_heads': 4,
            'dropout': 0.1,
        })

    elif model_type == 'hussain_hybrid':
        base.update({
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'd_model': 128,
            'num_heads': 4,
            'dropout': 0.2,
            'use_lr_scheduler': True,
            'use_early_stopping': True,
        })

    elif model_type == 'lightgbm':
        base.update({
            'num_leaves': 31,
            'learning_rate': 0.02,
            'max_depth': 8,
            'early_stopping_rounds': 200,
        })

    elif model_type == 'xgboost':
        base.update({
            'random_state': 16,
            'test_size': 0.20,
        })

    return base


# ── Health flags ───────────────────────────────────────────────────────────

def _compute_health_flags(
    all_metrics: Dict[int, dict],
    requested_horizons: List[int],
    smape_threshold: float = 30.0,
) -> dict:
    """Return health diagnostic flags derived from the experiment metrics."""
    completed = set(all_metrics.keys())
    requested = set(requested_horizons)
    missing = sorted(requested - completed)

    valid_smape = [
        v for h, m in all_metrics.items()
        if (v := m.get('SMAPE')) is not None and not (v != v)
    ]
    smape_ok = bool(valid_smape) and all(v < smape_threshold for v in valid_smape)

    smape_per_horizon = {
        f'{h}d': round(all_metrics[h].get('SMAPE', float('nan')), 2)
        for h in sorted(completed)
    }

    return {
        'smape_ok': smape_ok,
        'smape_per_horizon': smape_per_horizon,
        'horizons_completed': sorted(completed),
        'horizons_missing': missing,
        'all_horizons_completed': len(missing) == 0,
    }


# ── Single-case runner ─────────────────────────────────────────────────────

def run_single_case(
    case_config: CaseConfig,
    mlflow_tracking_uri: Optional[str] = None,
) -> dict:
    """Execute one (dataset × model × FE) experiment case and emit artifacts.

    Steps:

    1. Resolve dataset config (split date + date ranges).
    2. For tree models: one pipeline call with all horizons.
       For neural models: one pipeline call per horizon (look_back varies).
    3. Compute health flags from returned metrics.
    4. Write ``metadata.json`` and ``case.tex`` to ``case_config.case_dir``.

    Parameters
    ----------
    case_config : CaseConfig
        Specification of the case to run.
    mlflow_tracking_uri : str or None
        Forwarded to :func:`~src.forecast.pipeline.run_forecast_pipeline`.

    Returns
    -------
    dict
        The metadata dict written to ``metadata.json``.
    """
    from src.tfm.latex_generator import generate_case_latex

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    case_id = case_config.case_id
    logger.info("=" * 70)
    logger.info("START CASE: %s", case_id)
    logger.info("=" * 70)

    split_date_str, ranges = get_dataset_config(
        case_config.dataset, hussain=case_config.hussain
    )

    case_dir = case_config.case_dir
    case_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[int, dict] = {}
    look_back_map: Dict[int, int] = {}

    if case_config.model_type in _TREE_MODELS:
        # Single pipeline call handles all horizons internally
        params = _build_strategy_params_for_case(
            case_config.model_type,
            case_config.fe_params,
            case_config.li_forecast_horizons,
        )
        params.update(ranges)
        returned = run_forecast_pipeline(
            [case_config.dataset],
            params,
            split_date_str,
            model_type=case_config.model_type,
            output_dir=str(case_dir),
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        if returned:
            all_metrics.update(returned)
        for h in case_config.li_forecast_horizons:
            look_back_map[h] = 0  # tree models have no look_back warm-up

    else:
        # Neural models: one call per horizon with appropriate look_back
        for h in case_config.li_forecast_horizons:
            look_back = max(MIN_LOOK_BACK, h)  # floor at MIN_LOOK_BACK for all models incl. Hussain
            look_back_map[h] = look_back
            params = _build_strategy_params_for_case(
                case_config.model_type,
                case_config.fe_params,
                [h],
            )
            params['look_back'] = look_back
            params.update(ranges)
            returned = run_forecast_pipeline(
                [case_config.dataset],
                params,
                split_date_str,
                model_type=case_config.model_type,
                output_dir=str(case_dir),
                mlflow_tracking_uri=mlflow_tracking_uri,
            )
            if returned:
                all_metrics.update(returned)

    # Health flags
    health = _compute_health_flags(all_metrics, case_config.li_forecast_horizons)

    # Build metadata
    metadata = {
        'case_id': case_id,
        'dataset': case_config.dataset,
        'model': case_config.model_type,
        'fe_tag': case_config.fe_tag,
        'hussain_variant': case_config.hussain,
        'fe_config': case_config.fe_params,
        'split_date': split_date_str,
        'horizons': case_config.li_forecast_horizons,
        'look_back_per_horizon': look_back_map,
        'metrics': {f'{h}d': m for h, m in sorted(all_metrics.items())},
        'health': health,
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }

    # Write metadata.json
    meta_path = case_dir / 'metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.info("Written: %s", meta_path)

    # Generate and write case.tex
    latex = generate_case_latex(case_id, metadata, case_dir)
    tex_path = case_dir / 'case.tex'
    tex_path.write_text(latex, encoding='utf-8')
    logger.info("Written: %s", tex_path)

    logger.info("DONE CASE: %s  |  health=%s", case_id, health)
    return metadata


# ── Full-set runner ────────────────────────────────────────────────────────

def run_all_cases(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    fe_tags: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    output_base_dir: Optional[Path] = None,
    mlflow_tracking_uri: Optional[str] = None,
    skip_existing: bool = False,
) -> List[dict]:
    """Run the full combinatorial set of experiment cases.

    Parameters
    ----------
    datasets : list[str] or None
        Datasets to include.  Defaults to :data:`ALL_DATASETS`.
    models : list[str] or None
        Model types to include.  Defaults to :data:`ALL_MODELS`.
    fe_tags : list[str] or None
        Feature-engineering variant tags.  Defaults to :data:`ALL_FE_TAGS`.
    horizons : list[int] or None
        Forecast horizons in days.  Defaults to :data:`ALL_HORIZONS`.
    output_base_dir : Path or None
        Root directory for case artifacts.  Defaults to
        ``tfm/doc/vf/chapters/results/``.
    mlflow_tracking_uri : str or None
        Forwarded to :func:`run_single_case`.
    skip_existing : bool
        When ``True``, skip any case whose ``metadata.json`` already exists.

    Returns
    -------
    list[dict]
        List of metadata dicts, one per completed case.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    _datasets  = datasets  or ALL_DATASETS
    _models    = models    or ALL_MODELS
    _fe_tags   = fe_tags   or ALL_FE_TAGS
    _horizons  = horizons  or ALL_HORIZONS
    _base_dir  = output_base_dir or (_TFM_CHAPTERS_DIR / 'results')

    _base_dir.mkdir(parents=True, exist_ok=True)

    total = len(_datasets) * len(_models) * len(_fe_tags)
    logger.info(
        "run_all_cases: %d datasets × %d models × %d FE variants = %d cases",
        len(_datasets), len(_models), len(_fe_tags), total,
    )

    all_results: List[dict] = []
    failed: List[str] = []
    n = 0

    for dataset in _datasets:
        for model_type in _models:
            for fe_tag in _fe_tags:
                n += 1
                fe_params = FE_VARIANTS[fe_tag]
                case_config = CaseConfig(
                    dataset=dataset,
                    model_type=model_type,
                    fe_tag=fe_tag,
                    fe_params=fe_params,
                    li_forecast_horizons=_horizons,
                    output_base_dir=_base_dir,
                )
                if skip_existing and (case_config.case_dir / 'metadata.json').exists():
                    logger.info("[%d/%d] SKIP (existing): %s", n, total, case_config.case_id)
                    try:
                        with open(case_config.case_dir / 'metadata.json', encoding='utf-8') as fh:
                            all_results.append(json.load(fh))
                    except Exception:
                        pass
                    continue

                # Rolling FE tags are only run where a known demand-regime
                # shift makes recalibration worthwhile, and only for standard
                # neural models (Hussain reproductions have fixed hyperparams;
                # tree models have no incremental retraining support).
                if fe_tag in _ROLLING_FE_TAGS and (
                    dataset not in _ROLLING_DATASETS
                    or model_type in _HUSSAIN_MODELS
                    or model_type in _TREE_MODELS
                ):
                    logger.debug(
                        "[%d/%d] SKIP rolling FE '%s' for %s/%s (excluded).",
                        n, total, fe_tag, dataset, model_type,
                    )
                    continue

                logger.info("[%d/%d] Running: %s", n, total, case_config.case_id)
                try:
                    meta = run_single_case(case_config, mlflow_tracking_uri=mlflow_tracking_uri)
                    all_results.append(meta)
                except Exception as exc:
                    logger.error("FAILED case %s: %s", case_config.case_id, exc, exc_info=True)
                    failed.append(case_config.case_id)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("run_all_cases COMPLETE: %d/%d cases succeeded, %d failed",
                len(all_results), total, len(failed))
    if failed:
        logger.warning("Failed cases: %s", failed)

    # Print \input list for 04_resultados.tex
    print("\n% === \\input list for 04_resultados.tex ===")
    prev_dataset = None
    for meta in all_results:
        ds = meta.get('dataset', '')
        cid = meta.get('case_id', '')
        if ds != prev_dataset:
            print(f"\n% --- {ds} ---")
            prev_dataset = ds
        print(fr"\IfFileExists{{chapters/results/{cid}/case.tex}}{{\input{{chapters/results/{cid}/case}}}}{{}}")
    print("% ==========================================\n")

    return all_results


def old_main():
    
    datasets = ['Dundee', 
                #'ACN_Caltech', 
                # 'ACN_JPL', 
                #'ACN_Office001', 
                #'BeLib', 
                #'AMB_Barcelona'
                ]
    #datasets = ['ACN_Caltech', 'ACN_JPL']
    #datasets = ['Dundee']
    li_forecast_horizons = [1, 7, 30, 120]
    #datasets = ['ACN_JPL']
    #li_forecast_horizons = [30]
    split_date_str = "2023-09-01"
    n_trials = 5

    # MLflow tracking (set to None to disable)
    #MLFLOW_TRACKING_URI = "http://localhost:5000"
    MLFLOW_TRACKING_URI = None

    #optimize_lstm(datasets, n_trials, mlflow_tracking_uri=MLFLOW_TRACKING_URI)

    # tree based
    execute_lightgbm(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    execute_xgboost(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)

    execute_lstm(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    execute_transformer(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    execute_hybrid(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)

    # Hussain et al. (2025) article-variant models
    #execute_hussain_lstm(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    #execute_hussain_transformer(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    #execute_hussain_hybrid(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)


if __name__ == "__main__":
    MLFLOW_TRACKING_URI = None
    print("Running all cases with MLflow tracking URI:", MLFLOW_TRACKING_URI)
    run_all_cases()
