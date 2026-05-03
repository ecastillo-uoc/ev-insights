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
import os
import sys
import logging

import pandas as pd

# Add src to python path if not present
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def get_dataset_config(dataset_name, hussain=False):
    """Return (split_date_str, ranges_dict) for a given dataset.

    ranges_dict may contain 'train_range', 'test_range', 'zoom_range'.
    When a key is absent, run_forecast_pipeline uses the split_date fallback.
    """
    if hussain:
        # Hussain et al. article splits (COVID data intentionally included)
        _configs = {
            'ACN_JPL': {
                'split_date': '2021-01-01',
                'train_range': ('2018-09-01', '2020-08-05'),
                'test_range': ('2020-11-17', '2021-03-31'),
                'zoom_range': ('2021-01-01', '2021-03-31'),
            },
            'ACN_Caltech': {
                'split_date': '2021-01-01',
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
                'split_date': '2021-01-01',
                'train_range': ('2019-01-01', '2019-09-30'),
                'test_range': ('2019-10-01', '2020-03-01'),
                'zoom_range': ('2019-10-01', '2020-03-01'),
            },
            'ACN_Caltech': {
                'split_date': '2021-01-01',
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
                'split_date': '2021-01-01',
                'train_range': ('2018-09-01', '2020-08-05'),
                'test_range': ('2020-11-17', '2021-03-31'),
                'zoom_range': ('2021-01-01', '2021-03-31'),
            },
            'ACN_Caltech': {
                'split_date': '2021-01-01',
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
            look_back = max(MIN_LOOK_BACK, forecast_horizon)
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
            look_back = max(MIN_LOOK_BACK, forecast_horizon)
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
            look_back = max(MIN_LOOK_BACK, forecast_horizon)
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
#   * look_back = forecast_horizon (30, 120, 240) — vs. max(MIN_LOOK_BACK, h) for our models
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
    """LSTM with Hussain et al. hyperparams: look_back = forecast_horizon, dropout=0.2."""
    for ds in datasets:
        split_date_str, ranges = get_dataset_config(ds, hussain=True)
        for forecast_horizon in li_forecast_horizons:
            strategy_params = {
                'epochs': 100,
                'batch_size': 32,
                'look_back': forecast_horizon,       # Article: look_back = prediction period
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
                'look_back': forecast_horizon,       # Article: look_back = prediction period
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
    """Hybrid LSTM-Transformer with Hussain et al. hyperparams: look_back = forecast_horizon, dropout=0.2."""
    for ds in datasets:
        split_date_str, ranges = get_dataset_config(ds, hussain=True)
        for forecast_horizon in li_forecast_horizons:
            strategy_params = {
                'epochs': 100,
                'batch_size': 32,
                'look_back': forecast_horizon,       # Article: look_back = prediction period
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


if __name__ == "__main__":
    
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
