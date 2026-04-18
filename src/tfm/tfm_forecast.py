import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sys
import logging
from datetime import timedelta

# Add src to python path if not present
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Own modules
from src.utils.console import Colors

# forecast
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from src.forecast.strategies import (
    LSTMModelStrategy, TransformerModelStrategy, HussainTransformerModelStrategy,
    LightGBMModelStrategy, XGBoostModelStrategy, HybridTransformerLSTMModelStrategy,
    HussainHybridModelStrategy,
)
from src.forecast.model_persistence import save_model, load_model
from src.forecast.strategies.utils_ts import smape

# MLflow (optional, for experiment tracking)
try:
    import mlflow
    import joblib
    import tempfile
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

# Optimization
import optuna
from src.tfm.param_optimization import objective_lstm, objective_transformer

# Data preparation
from src.tfm.tfm_data_fetcher import fetch_daily_energy_for_forecast
from src.tfm.tfm_constants import COVID_START, COVID_END

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def plot_train_test_split(train, test, dataset_name, forecast_horizon, zoom_range=None, model_name=None):
    """Plot the target variable over time with distinct colours for train and test periods."""
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['y'], label='Train', linewidth=1.5, color='#1f77b4')
    plt.plot(test.index, test['y'], label='Test (Actual)', linewidth=1.5, color='#ff7f0e')

    # Vertical line at the train/test boundary
    split_date = train.index.max()
    plt.axvline(x=split_date, color='grey', linestyle='--', linewidth=1, label='Train / Test split')

    title = f'Train vs Test Split for {dataset_name}'
    if model_name:
        title += f'\nModel: {model_name}'
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Energy delivered (kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save full plot
    os.makedirs('output_plots', exist_ok=True)
    suffix = f'_{model_name}' if model_name else ''
    plot_path = f'output_plots/{dataset_name}_train_test_split{suffix}_{forecast_horizon}days.png'
    plt.savefig(plot_path)
    logging.info(f"Train/Test split plot saved to {plot_path}")

    # Save zoom plot if range is provided
    plot_path_zoom = None
    if zoom_range:
        plt.xlim(pd.to_datetime(zoom_range[0]), pd.to_datetime(zoom_range[1]))
        plot_path_zoom = f'output_plots/{dataset_name}_train_test_split{suffix}_{forecast_horizon}days_zoom.png'
        plt.savefig(plot_path_zoom)
        logging.info(f"Train/Test split zoom plot saved to {plot_path_zoom}")

    plt.close()
    return plot_path




def plot_test_vs_predict(test, predictions, dataset_name, model_name, forecast_horizon):
    """Plot the test data (actual) against predictions."""
    min_len = min(len(test), len(predictions))
    if min_len < len(test) or min_len < len(predictions):
        logging.warning(f"Length mismatch: test={len(test)}, predictions={len(predictions)}. Truncating to {min_len}.")
    test = test.iloc[:min_len]
    predictions = predictions[:min_len]
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['y'], label='Test (Actual)', linewidth=1.5)
    plt.plot(test.index, predictions, label='Predictions', linestyle='--', linewidth=1.5, color='red')
    plt.title(f'Actual vs Predictions for {dataset_name} ({forecast_horizon} days)\nModel: {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Energy Demand (kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save plot
    os.makedirs('output_plots', exist_ok=True)
    plot_path = f'output_plots/{dataset_name}_actual_vs_predict_{model_name}_{forecast_horizon}days.png'
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Actual vs Predict plot saved to {plot_path}")

def run_forecast_pipeline(datasets, strategy_params, split_date_str, model_type="lstm", model_suffix="",
                          mlflow_tracking_uri=None):
    
    models_dir = 'output_models'
    os.makedirs(models_dir, exist_ok=True)
    li_forecast_horizons = strategy_params.get('li_forecast_horizons', [1])

    results_summary = []

    for dataset in datasets:
        logging.info(f"--- Processing Dataset: {dataset} ---")
        is_hussain = model_type.startswith("hussain_")
        # COVID exclusion is governed globally by EXCLUDE_COVID_DATA for ALL
        # strategies — including Hussain variants.
        df = fetch_daily_energy_for_forecast(dataset)
        
        if df.empty:
            logging.warning(f"No data found for {dataset}. Skipping.")
            continue

        for forecast_horizon in li_forecast_horizons:
            logging.info(f"Prediction window: {forecast_horizon} days")

            # Make forecast_horizon available to the strategy layer
            strategy_params['forecast_horizon'] = forecast_horizon
            
            # 1. Split train/test
            if len(df) <= forecast_horizon:
                logging.warning(f"Not enough data for test size {forecast_horizon}. Skipping.")
                continue
                

            # Apply the explicit test_range boundaries for predictions if provided, else use default split_date rules
            split_date = pd.to_datetime(split_date_str)
            test_range = strategy_params.get('test_range', None)
            if test_range:
                test_start, test_end = test_range
                test_mask = pd.Series(True, index=df.index)
                if test_start:
                    test_mask = test_mask & (df.index >= pd.to_datetime(test_start))
                else:
                    test_mask = test_mask & (df.index > split_date) # Fallback start
                if test_end:
                    test_mask = test_mask & (df.index <= pd.to_datetime(test_end))
                test_df = df[test_mask].copy()
            else:
                test_df = df[df.index > split_date].copy()
            
            # Local copies for plotting and the prediction baseline
            train_range = strategy_params.get('train_range', None)
            if train_range:
                train_start, train_end = train_range
                train_mask = pd.Series(True, index=df.index)
                if train_start:
                    train_mask = train_mask & (df.index >= pd.to_datetime(train_start))
                if train_end:
                    train_mask = train_mask & (df.index <= pd.to_datetime(train_end))
                else: 
                    train_mask = train_mask & (df.index <= split_date) # Fallback end
                train_df = df[train_mask].copy()
            else:
                train_df = df[df.index <= split_date].copy()
            
            # 2. Train model using appropriate ModelStrategy
            if model_type == "transformer":
                strategy = TransformerModelStrategy()
                model_name_prefix = f"transformer_{forecast_horizon}d"
                
                df_model = df.copy()
                df_model['dataset_name'] = dataset
                df_model.attrs['transformer_params'] = strategy_params
                
                train_df['dataset_name'] = dataset
                train_df.attrs['transformer_params'] = strategy_params
            elif model_type == "lightgbm":
                strategy = LightGBMModelStrategy()
                model_name_prefix = f"lightgbm_{forecast_horizon}d"
                
                df_model = df.copy()
                df_model['dataset_name'] = dataset
                df_model.attrs['lgbm_params'] = strategy_params
                
                train_df['dataset_name'] = dataset
                train_df.attrs['lgbm_params'] = strategy_params
            elif model_type == "xgboost":
                strategy = XGBoostModelStrategy()
                model_name_prefix = f"xgboost_{forecast_horizon}d"
                
                df_model = df.copy()
                df_model['dataset_name'] = dataset
                df_model.attrs['xgb_params'] = strategy_params
                
                train_df['dataset_name'] = dataset
                train_df.attrs['xgb_params'] = strategy_params
            elif model_type == "hybrid":
                strategy = HybridTransformerLSTMModelStrategy()
                model_name_prefix = f"hybrid_{forecast_horizon}d"
                
                df_model = df.copy()
                df_model['dataset_name'] = dataset
                df_model.attrs['hybrid_params'] = strategy_params
                
                train_df['dataset_name'] = dataset
                train_df.attrs['hybrid_params'] = strategy_params
            elif model_type == "hussain_lstm":
                strategy = LSTMModelStrategy()
                model_name_prefix = f"hussain_lstm_{forecast_horizon}d"
                
                df_model = df.copy()
                df_model['dataset_name'] = dataset
                df_model.attrs['lstm_params'] = strategy_params
                
                train_df['dataset_name'] = dataset
                train_df.attrs['lstm_params'] = strategy_params
            elif model_type == "hussain_transformer":
                strategy = HussainTransformerModelStrategy()
                model_name_prefix = f"hussain_transformer_{forecast_horizon}d"
                
                df_model = df.copy()
                df_model['dataset_name'] = dataset
                df_model.attrs['hussain_transformer_params'] = strategy_params
                
                train_df['dataset_name'] = dataset
                train_df.attrs['hussain_transformer_params'] = strategy_params
            elif model_type == "hussain_hybrid":
                strategy = HussainHybridModelStrategy()
                model_name_prefix = f"hussain_hybrid_{forecast_horizon}d"
                
                df_model = df.copy()
                df_model['dataset_name'] = dataset
                df_model.attrs['hussain_hybrid_params'] = strategy_params
                
                train_df['dataset_name'] = dataset
                train_df.attrs['hussain_hybrid_params'] = strategy_params
            else:
                strategy = LSTMModelStrategy()
                model_name_prefix = f"lstm_{forecast_horizon}d"
                
                df_model = df.copy()
                df_model['dataset_name'] = dataset
                df_model.attrs['lstm_params'] = strategy_params
                
                train_df['dataset_name'] = dataset
                train_df.attrs['lstm_params'] = strategy_params
            
            feature_cols = []
            if model_type in ["xgboost", "lightgbm"]:
                for i in range(1, forecast_horizon + 1):
                    col_name = f'lag_{i}'
                    df_model[col_name] = df_model['y'].shift(i)
                    train_df[col_name] = train_df['y'].shift(i)
                    feature_cols.append(col_name)
                df_model = df_model.dropna()
                train_df = train_df.dropna()

            logging.info(f"Training model with split_date={split_date_str}...")
            trained_models_dict = strategy.train(
                df=df_model, 
                feature_columns=feature_cols, 
                target_column='y', 
                dataset_names=[dataset], 
                model_name_prefix=model_name_prefix,
                split_date=split_date_str
            )
            trained_model_objects = trained_models_dict['train'][f"{model_name_prefix}_{dataset}"]['model']
            
            # 3. Store the trained model
            logging.info("Saving model...")
            model_name = f"{model_name_prefix}_{dataset}{model_suffix}"
            save_model(model_name, trained_models_dict, models_dir)
            
            # 4. Predict
            logging.info("Generating predictions...")
            
            # For LSTM/Transformer models we use backtest mode (rolling one-step on test data)
            # For tree-based models we predict directly on the test data which already has the lag features
            # For Hussain models we use SCHEDULE mode (multi-step recursive) to match
            # the article methodology: feed last look_back days of training data and
            # recursively predict forecast_horizon days ahead (visible in Figs 9-26
            # where x-axis = "Number of Days" 0..N).
            if model_type in ["xgboost", "lightgbm"]:
                predict_df = test_df.copy()
                predict_df['dataset_name'] = dataset
                # Ensure lag features are computed properly on test_df by relying on the full df_model shifts
                for i in range(1, forecast_horizon + 1):
                    predict_df[f'lag_{i}'] = df_model.loc[predict_df.index, f'lag_{i}']
                predict_df = predict_df.dropna()
                
                predict_mode = None
            elif is_hussain:
                # Direct multi-step: seed from the first look_back days of
                # test data so that predictions fall inside the test range
                # (starting at test_start + look_back).  We cannot seed from
                # training data because the predicted dates would land in the
                # COVID gap where no valid data exists.
                look_back = strategy_params.get('look_back', forecast_horizon)
                predict_df = test_df.iloc[:look_back].copy()
                predict_df['dataset_name'] = dataset
                predict_mode = 'direct_multistep'
            else:
                predict_df = test_df.copy()
                predict_df['dataset_name'] = dataset
                predict_mode = None  # Triggers backtest/validate mode

            predict_dict = strategy.predict(
                df=predict_df, 
                feature_columns=feature_cols, 
                target_column='y', 
                model_objects=trained_model_objects, 
                context_date=None, 
                dataset_names=[dataset], 
                submode=predict_mode
            )
            predictions = predict_dict['predict']['values']
            predict_dates = predict_dict['predict'].get('dates')
            predict_actuals = predict_dict['predict'].get('actuals')
            
            # 5. Provide metrics
            # Align predictions and actuals using the dates returned by the
            # predict method so that look_back / forecast_horizon offsets are
            # properly accounted for.  The first look_back days of test data
            # are consumed as seed context and have no corresponding predictions.
            preds_array = np.array(predictions)
            
            if predict_dates is not None:
                predict_dates_idx = pd.to_datetime(predict_dates)
                
                if predict_actuals is not None:
                    # Backtest mode: actuals already aligned with predictions
                    actuals_array = np.array(predict_actuals)
                else:
                    # Direct multistep / schedule: look up actuals from full df
                    available_mask = predict_dates_idx.isin(df.index)
                    common_dates = predict_dates_idx[available_mask]
                    actuals_array = df.loc[common_dates, 'y'].values
                    preds_array = preds_array[available_mask]
                    predict_dates_idx = common_dates
                
                min_len = min(len(preds_array), len(actuals_array))
                a, p = actuals_array[:min_len], preds_array[:min_len]
                # DataFrame with correct prediction-aligned dates for plotting
                plot_actuals_df = pd.DataFrame({'y': a}, index=predict_dates_idx[:min_len])
            else:
                # Tree-based models: no dates in predict output — fall back to test_df
                actuals_array = test_df['y'].values
                min_len = min(len(preds_array), len(actuals_array))
                a, p = actuals_array[:min_len], preds_array[:min_len]
                plot_actuals_df = test_df.iloc[:min_len]
            mse_val = mean_squared_error(a, p)
            metrics = {
                'MSE': mse_val,
                # RMSE is included because Hussain et al. (2025) appear to
                # report RMSE labelled as "MSE" — their reported "MSE" values
                # are only slightly above the corresponding MAE, which is
                # consistent with RMSE (√MSE) but not raw MSE.
                'RMSE': math.sqrt(mse_val),
                'MAE': mean_absolute_error(a, p),
                'MAPE': mean_absolute_percentage_error(a, p),
                'SMAPE': smape(p, a)
            }
            logging.info(f"Metrics for {dataset} ({forecast_horizon} days): {metrics}")
            
            results_summary.append({
                'dataset': dataset,
                'forecast_horizon': forecast_horizon,
                **metrics
            })
            
            # 6. Plot real vs predictions
            zoom_range = strategy_params.get('zoom_range', None)
            plot_path_split = plot_train_test_split(
                train_df, test_df, dataset, forecast_horizon,
                zoom_range=zoom_range, model_name=model_name
            )

            plot_path_predict = f'output_plots/{dataset}_actual_vs_predict_{model_name}_{forecast_horizon}days.png'
            plot_test_vs_predict(plot_actuals_df, p, dataset, model_name, forecast_horizon)

            # 7. Log to MLflow (optional)
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
                        # Log hyperparams (filter to serialisable scalar values)
                        for k, v in strategy_params.items():
                            if isinstance(v, (int, float, str, bool)):
                                mlflow.log_param(k, v)
                        mlflow.log_param('model_type', model_type)
                        mlflow.log_param('split_date', split_date_str)
                        mlflow.log_param('forecast_horizon', forecast_horizon)

                        # Log metrics
                        mlflow.log_metrics(metrics)

                        # Log model artifacts
                        train_output = trained_models_dict['train'].get(f"{model_name_prefix}_{dataset}", {})
                        model_obj = train_output.get('model')
                        if isinstance(model_obj, dict) and 'keras_model' in model_obj:
                            mlflow.keras.log_model(model_obj['keras_model'], artifact_path=f"{model_type}_model",
                                                   registered_model_name=model_name)
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
                                    mlflow.lightgbm.log_model(model_obj, artifact_path=f"{model_type}_model",
                                                              registered_model_name=model_name)
                                elif model_type == 'xgboost':
                                    mlflow.xgboost.log_model(model_obj, artifact_path=f"{model_type}_model",
                                                             registered_model_name=model_name)
                            except Exception as model_log_err:
                                logging.warning(f"Could not log model to MLflow: {model_log_err}")

                        # Log plot artifacts
                        if os.path.exists(plot_path_predict):
                            mlflow.log_artifact(plot_path_predict, artifact_path="plots")
                        if os.path.exists(plot_path_split):
                            mlflow.log_artifact(plot_path_split, artifact_path="plots")

                    logging.info(f"MLflow run logged for {model_name}")
                except Exception as mlflow_err:
                    logging.warning(f"MLflow logging failed (non-blocking): {mlflow_err}")
            
    summary_df = pd.DataFrame(results_summary)
    logging.info("\\nFinal Benchmark Summary:\\n" + summary_df.to_string())
    summary_df.to_csv('forecast_metrics_summary.csv', index=False)
    logging.info("Metrics saved to forecast_metrics_summary.csv")

def execute_lstm(datsets, li_forecast_horizons, mlflow_tracking_uri=None): 
    # fetch and plot
    split_date_str = "2021-01-01"
    strategy_params_lstm = {
        'epochs': 100, 
        'batch_size': 32,
        'look_back': 14,           # 2 weeks look back
        'li_forecast_horizons': li_forecast_horizons,
        'learning_rate': 0.001,    # Default used if missing
        'dropout_rate': 0.2,       # Default used if missing
        'activation': 'relu',      # Default used if missing
    }

    strategy_jpl = {
        'train_range': ('2019-01-01', '2019-09-30'), # Optional train
        'test_range': ('2019-10-01', '2020-03-01'), # Optional test
        'zoom_range': ('2019-10-01', '2020-03-01') # Concrete daterange for zoomed plot
    }

    # Merge base LSTM strategy params with the JPL specific overrides
    strategy_params_lstm_jpl = {**strategy_params_lstm, **strategy_jpl}

    # run_forecast_pipeline(['ACN_JPL'], strategy_params_lstm_jpl, split_date_str)


    strategy_caltech = {
        'train_range': ('2019-01-01', '2019-06-30'), # Optional train
        'test_range': ('2019-07-01', '2019-12-15'), # Optional test
        'zoom_range': ('2019-07-01', '2019-12-15') # Concrete daterange for zoomed plot
    }

    # Merge base LSTM strategy params with the JPL specific overrides
    strategy_params_lstm_caltech = {**strategy_params_lstm, **strategy_caltech}

    run_forecast_pipeline(['ACN_Caltech'], strategy_params_lstm_caltech, split_date_str, mlflow_tracking_uri=mlflow_tracking_uri)

def execute_transformer(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    split_date_str = "2021-01-01"
    strategy_params_transformer = {
        'epochs': 100, 
        'batch_size': 32,
        'look_back': 14,
        'li_forecast_horizons': li_forecast_horizons,
        'learning_rate': 0.001,
        'head_size': 128,
        'num_heads': 4,
        'ff_dim': 4,
        'num_transformer_blocks': 2,
        'dropout': 0.1,
        'mlp_dropout': 0.1,
    }

    strategy_jpl = {
        'train_range': ('2019-01-01', '2019-09-30'), 
        'test_range': ('2019-10-01', '2020-03-01'), 
        'zoom_range': ('2019-10-01', '2020-03-01') 
    }

    strategy_params_transformer_jpl = {**strategy_params_transformer, **strategy_jpl}

    # Only run if ACN_JPL is in datasets
    if 'ACN_JPL' in datasets:
        run_forecast_pipeline(['ACN_JPL'], strategy_params_transformer_jpl, split_date_str, model_type="transformer", mlflow_tracking_uri=mlflow_tracking_uri)

    strategy_caltech = {
        'train_range': ('2019-01-01', '2019-06-30'),
        'test_range': ('2019-07-01', '2019-12-15'), 
        'zoom_range': ('2019-07-01', '2019-12-15') 
    }

    strategy_params_transformer_caltech = {**strategy_params_transformer, **strategy_caltech}

    # Only run if ACN_Caltech is in datasets
    if 'ACN_Caltech' in datasets:
        run_forecast_pipeline(['ACN_Caltech'], strategy_params_transformer_caltech, split_date_str, model_type="transformer", mlflow_tracking_uri=mlflow_tracking_uri)

def execute_lightgbm(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    split_date_str = "2021-01-01"
    strategy_params_lgbm = {
        'li_forecast_horizons': li_forecast_horizons,
        # Default LightGBM specific params can be passed here
        'num_leaves': 10,
        'learning_rate': 0.02,
        'max_depth': 5,
        'early_stopping_rounds': 200,
    }

    strategy_jpl = {
        'train_range': ('2019-01-01', '2019-09-30'), 
        'test_range': ('2019-10-01', '2020-03-01'), 
        'zoom_range': ('2019-10-01', '2020-03-01') 
    }

    strategy_params_lgbm_jpl = {**strategy_params_lgbm, **strategy_jpl}

    if 'ACN_JPL' in datasets:
        run_forecast_pipeline(['ACN_JPL'], strategy_params_lgbm_jpl, split_date_str, model_type="lightgbm", mlflow_tracking_uri=mlflow_tracking_uri)

    strategy_caltech = {
        'train_range': ('2019-01-01', '2019-06-30'),
        'test_range': ('2019-07-01', '2019-12-15'), 
        'zoom_range': ('2019-07-01', '2019-12-15') 
    }

    strategy_params_lgbm_caltech = {**strategy_params_lgbm, **strategy_caltech}

    if 'ACN_Caltech' in datasets:
        run_forecast_pipeline(['ACN_Caltech'], strategy_params_lgbm_caltech, split_date_str, model_type="lightgbm", mlflow_tracking_uri=mlflow_tracking_uri)

def execute_xgboost(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    split_date_str = "2021-01-01"
    strategy_params_xgb = {
        'li_forecast_horizons': li_forecast_horizons,
        'random_state': 16,
        'test_size': 0.20
    }

    strategy_jpl = {
        'train_range': ('2019-01-01', '2019-09-30'), 
        'test_range': ('2019-10-01', '2020-03-01'), 
        'zoom_range': ('2019-10-01', '2020-03-01') 
    }

    strategy_params_xgb_jpl = {**strategy_params_xgb, **strategy_jpl}

    if 'ACN_JPL' in datasets:
        run_forecast_pipeline(['ACN_JPL'], strategy_params_xgb_jpl, split_date_str, model_type="xgboost", mlflow_tracking_uri=mlflow_tracking_uri)

    strategy_caltech = {
        'train_range': ('2019-01-01', '2019-06-30'),
        'test_range': ('2019-07-01', '2019-12-15'), 
        'zoom_range': ('2019-07-01', '2019-12-15') 
    }

    strategy_params_xgb_caltech = {**strategy_params_xgb, **strategy_caltech}

    if 'ACN_Caltech' in datasets:
        run_forecast_pipeline(['ACN_Caltech'], strategy_params_xgb_caltech, split_date_str, model_type="xgboost", mlflow_tracking_uri=mlflow_tracking_uri)


def execute_hybrid(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    split_date_str = "2021-01-01"
    strategy_params_hybrid = {
        'epochs': 100, 
        'batch_size': 32,
        'look_back': 14,
        'li_forecast_horizons': li_forecast_horizons,
        'learning_rate': 0.001,
        'd_model': 128,
        'num_heads': 4,
        'dropout': 0.1,
    }

    strategy_jpl = {
        'train_range': ('2019-01-01', '2019-09-30'), 
        'test_range': ('2019-10-01', '2020-03-01'), 
        'zoom_range': ('2019-10-01', '2020-03-01') 
    }

    strategy_params_hybrid_jpl = {**strategy_params_hybrid, **strategy_jpl}

    if 'ACN_JPL' in datasets:
        run_forecast_pipeline(['ACN_JPL'], strategy_params_hybrid_jpl, split_date_str, model_type="hybrid", mlflow_tracking_uri=mlflow_tracking_uri)

    strategy_caltech = {
        'train_range': ('2019-01-01', '2019-06-30'),
        'test_range': ('2019-07-01', '2019-12-15'), 
        'zoom_range': ('2019-07-01', '2019-12-15') 
    }

    strategy_params_hybrid_caltech = {**strategy_params_hybrid, **strategy_caltech}

    if 'ACN_Caltech' in datasets:
        run_forecast_pipeline(['ACN_Caltech'], strategy_params_hybrid_caltech, split_date_str, model_type="hybrid", mlflow_tracking_uri=mlflow_tracking_uri)


# =============================================================================
# Hussain et al. (2025) article-variant models
# --------------------------------------------------------------------------
# These functions reproduce the architectures and hyperparameters described in:
#   Hussain, A. et al. "Charging stations demand forecasting using LSTM based
#   hybrid transformer model." Sci Rep 15, 13555 (2025).
#
# Key differences from our default models:
#   * look_back = forecast_horizon (30, 120, 240) — not a fixed 14
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
    split_date_str = "2021-01-01"

    # Article splits: train ~24 months (Sep 2018 – Aug 2020), test ~7 months (Sep 2020 – Mar 2021).
    # COVID period is intentionally included, consistent with the original paper.
    strategy_article = {
        'train_range': ('2018-09-01', '2020-08-05'),
        'test_range': ('2020-11-17', '2021-03-31'),
        #'zoom_range': ('2020-09-01', '2021-03-31'),
        'zoom_range': ('2021-01-01', '2021-03-31'),
    }

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
            'predict_mode': 'direct_multistep',
        }

        if 'ACN_JPL' in datasets:
            params_jpl = {**strategy_params, **strategy_article}
            run_forecast_pipeline(['ACN_JPL'], params_jpl, split_date_str,
                                  model_type="hussain_lstm", mlflow_tracking_uri=mlflow_tracking_uri)

        if 'ACN_Caltech' in datasets:
            params_caltech = {**strategy_params, **strategy_article}
            run_forecast_pipeline(['ACN_Caltech'], params_caltech, split_date_str,
                                  model_type="hussain_lstm", mlflow_tracking_uri=mlflow_tracking_uri)


def execute_hussain_transformer(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    """Simplified Transformer from Hussain et al.: Dense→MHA→GAP→Dense(1)."""
    split_date_str = "2021-01-01"

    # Article splits: train ~24 months (Sep 2018 – Aug 2020), test ~7 months (Sep 2020 – Mar 2021).
    # COVID period is intentionally included, consistent with the original paper.
    strategy_article = {
        'train_range': ('2018-09-01', '2020-08-05'),
        'test_range': ('2020-11-17', '2021-03-31'),
        #'zoom_range': ('2020-09-01', '2021-03-31'),
        'zoom_range': ('2021-01-01', '2021-03-31'),
    }

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
            'predict_mode': 'direct_multistep',
        }

        if 'ACN_JPL' in datasets:
            params_jpl = {**strategy_params, **strategy_article}
            run_forecast_pipeline(['ACN_JPL'], params_jpl, split_date_str,
                                  model_type="hussain_transformer", mlflow_tracking_uri=mlflow_tracking_uri)

        if 'ACN_Caltech' in datasets:
            params_caltech = {**strategy_params, **strategy_article}
            run_forecast_pipeline(['ACN_Caltech'], params_caltech, split_date_str,
                                  model_type="hussain_transformer", mlflow_tracking_uri=mlflow_tracking_uri)


def execute_hussain_hybrid(datasets, li_forecast_horizons, mlflow_tracking_uri=None):
    """Hybrid LSTM-Transformer with Hussain et al. hyperparams: look_back = forecast_horizon, dropout=0.2."""
    split_date_str = "2021-01-01"

    # Article splits: train ~24 months (Sep 2018 – Aug 2020), test ~7 months (Sep 2020 – Mar 2021).
    # COVID period is intentionally included, consistent with the original paper.
    strategy_article = {
        'train_range': ('2018-09-01', '2020-08-05'),
        'test_range': ('2020-11-17', '2021-03-31'),
        #'zoom_range': ('2020-09-01', '2021-03-31'),
        'zoom_range': ('2021-01-01', '2021-03-31'),
    }

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
            'predict_mode': 'direct_multistep',
        }

        if 'ACN_JPL' in datasets:
            params_jpl = {**strategy_params, **strategy_article}
            run_forecast_pipeline(['ACN_JPL'], params_jpl, split_date_str,
                                  model_type="hussain_hybrid", mlflow_tracking_uri=mlflow_tracking_uri)

        if 'ACN_Caltech' in datasets:
            params_caltech = {**strategy_params, **strategy_article}
            run_forecast_pipeline(['ACN_Caltech'], params_caltech, split_date_str,
                                  model_type="hussain_hybrid", mlflow_tracking_uri=mlflow_tracking_uri)


def optimize_lstm(datasets, n_trials, mlflow_tracking_uri=None):
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
    
    # li_ds = ['ACN_Caltech', 'ACN_JPL', 'ACN_Office001', 'BeLib', 'AMB_Barcelona']
    # datasets = ['ACN_Caltech', 'ACN_JPL']
    # li_forecast_horizons = [30, 120, 240]
    datasets = ['ACN_JPL']
    li_forecast_horizons = [30]
    split_date_str = "2021-01-01"
    n_trials = 5

    # MLflow tracking (set to None to disable)
    #MLFLOW_TRACKING_URI = "http://localhost:5000"
    MLFLOW_TRACKING_URI = None

    #optimize_lstm(datasets, n_trials, mlflow_tracking_uri=MLFLOW_TRACKING_URI)

    # tree based
    #execute_lightgbm(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    #execute_xgboost(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)

    #execute_lstm(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    #execute_transformer(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    # execute_hybrid(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)

    # Hussain et al. (2025) article-variant models
    execute_hussain_lstm(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    execute_hussain_transformer(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    execute_hussain_hybrid(datasets, li_forecast_horizons, mlflow_tracking_uri=MLFLOW_TRACKING_URI)
