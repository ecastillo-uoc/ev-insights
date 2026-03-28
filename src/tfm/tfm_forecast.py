import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sys
import logging
from datetime import timedelta

# Add src to python path if not present
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from src.forecast.forecast_implementations import LSTMModelStrategy
from src.forecast.model_persistence import save_model, load_model
from src.forecast.strategies.utils_ts import smape

from src.tfm.tfm_data_fetcher import fetch_daily_energy_for_forecast
from src.tfm.tfm_constants import COVID_START, COVID_END

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_train_test_split(train, test, dataset_name, test_size, zoom_range=None):
    """Plot the real training data and test data split."""
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['y'], label='Train', linewidth=1.5)
    plt.plot(test.index, test['y'], label='Test (Actual)', linewidth=1.5)
    plt.title(f'Train vs Test Split for {dataset_name}')
    plt.xlabel('Date')
    plt.ylabel('Energy delivered (kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save full plot
    os.makedirs('output_plots', exist_ok=True)
    plot_path = f'output_plots/{dataset_name}_train_test_split_{test_size}days.png'
    plt.savefig(plot_path)
    logging.info(f"Train/Test split plot saved to {plot_path}")

    # Save zoom plot if range is provided
    if zoom_range:
        plt.xlim(pd.to_datetime(zoom_range[0]), pd.to_datetime(zoom_range[1]))
        plot_path_zoom = f'output_plots/train_test_split_{dataset_name}_{test_size}days_zoom.png'
        plt.savefig(plot_path_zoom)
        logging.info(f"Train/Test split zoom plot saved to {plot_path_zoom}")

    plt.close()




def plot_test_vs_predict(test, predictions, dataset_name, model_name, test_size):
    """Plot the test data (actual) against predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['y'], label='Test (Actual)', linewidth=1.5)
    plt.plot(test.index, predictions, label='Predictions', linestyle='--', linewidth=1.5, color='red')
    plt.title(f'Actual vs Predictions for {dataset_name} ({test_size} days)\nModel: {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Energy Demand (kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save plot
    os.makedirs('output_plots', exist_ok=True)
    plot_path = f'output_plots/{dataset_name}_actual_vs_predict_{model_name}_{test_size}days.png'
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Actual vs Predict plot saved to {plot_path}")

def run_forecast_pipeline(datasets, strategy_params, split_date_str):
    
    models_dir = 'output_models'
    os.makedirs(models_dir, exist_ok=True)
    prediction_lag_days = strategy_params.get('prediction_lag_days', [1])

    results_summary = []

    for dataset in datasets:
        logging.info(f"--- Processing Dataset: {dataset} ---")
        df = fetch_daily_energy_for_forecast(dataset)
        
        if df.empty:
            logging.warning(f"No data found for {dataset}. Skipping.")
            continue

        for lag_size_days in prediction_lag_days:
            logging.info(f"Prediction window: {lag_size_days} days")
            
            # 1. Split train/test
            if len(df) <= lag_size_days:
                logging.warning(f"Not enough data for test size {lag_size_days}. Skipping.")
                continue
                
            # Apply the explicit test_range boundaries for predictions if provided, else use default split_date rules
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
            
            # 2. Train model using LSTMModelStrategy
            strategy = LSTMModelStrategy()
            model_name_prefix = f"lstm_{lag_size_days}d"
            
            # Parameters dictionary (if omitted, falls back to defaults natively inside train)

            
            # Format full df to pass to train() so it applies the split_date internally
            df_model = df.copy()
            df_model['dataset_name'] = dataset
            df_model.attrs['lstm_params'] = strategy_params
            
            # Format train_df for predict() call later
            train_df['dataset_name'] = dataset
            train_df.attrs['lstm_params'] = strategy_params
            
            logging.info(f"Training model with split_date={split_date_str}...")
            trained_models_dict = strategy.train(
                df=df_model, 
                feature_columns=[], 
                target_column='y', 
                dataset_names=[dataset], 
                model_name_prefix=model_name_prefix,
                split_date=split_date_str
            )
            trained_model_objects = trained_models_dict['train'][f"{model_name_prefix}_{dataset}"]['model']
            
            # 3. Store the trained model
            logging.info("Saving model...")
            model_name = f"{model_name_prefix}_{dataset}"
            save_model(model_name, trained_models_dict, models_dir)
            
            # 4. Predict
            logging.info("Generating predictions...")
            predict_dict = strategy.predict(
                df=train_df, 
                feature_columns=[], 
                target_column='y', 
                model_objects=trained_model_objects, 
                context_date=None, 
                dataset_names=[dataset], 
                submode='schedule'
            )
            predictions = predict_dict['predict']['values']
            
            # 5. Provide metrics
            preds_array = np.array(predictions)
            actuals_array = test_df['y'].values
            
            # Handling lengths
            min_len = min(len(preds_array), len(actuals_array))
            a, p = actuals_array[:min_len], preds_array[:min_len]
            metrics = {
                'MSE': mean_squared_error(a, p),
                'MAE': mean_absolute_error(a, p),
                'MAPE': mean_absolute_percentage_error(a, p),
                'SMAPE': smape(p, a)
            }
            logging.info(f"Metrics for {dataset} ({lag_size_days} days): {metrics}")
            
            results_summary.append({
                'dataset': dataset,
                'lag_size_days': lag_size_days,
                **metrics
            })
            
            # 6. Plot real vs predictions
            zoom_range = strategy_params.get('zoom_range', None)
            plot_train_test_split(train_df, test_df, dataset, lag_size_days, zoom_range=zoom_range)
            plot_test_vs_predict(test_df.iloc[:min_len], preds_array[:min_len], dataset, model_name, lag_size_days)
            
    summary_df = pd.DataFrame(results_summary)
    logging.info("\\nFinal Benchmark Summary:\\n" + summary_df.to_string())
    summary_df.to_csv('forecast_metrics_summary.csv', index=False)
    logging.info("Metrics saved to forecast_metrics_summary.csv")

if __name__ == "__main__":

    # fetch and plot
    # li_ds = ['ACN_Caltech', 'ACN_JPL', 'ACN_Office001', 'BeLib', 'AMB_Barcelona']
    datasets = ['ACN_Caltech', 'ACN_JPL']
    split_date_str = "2021-01-01"
    prediction_lag_days = [30, 120, 240]
    strategy_params_lstm = {
        'epochs': 100, 
        'batch_size': 32,
        'look_back': 14,           # 2 weeks look back
        'prediction_lag_days': prediction_lag_days,
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

    run_forecast_pipeline(['ACN_Caltech'], strategy_params_lstm_caltech, split_date_str)
