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

from src.forecast.forecast_implementations import LSTMModelStrategy
from src.forecast.model_persistence import save_model, load_model
from src.forecast.strategies.utils_ts import smape
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.tfm.tfm_data_fetcher import fetch_daily_energy_for_forecast

from tfm_constants import COVID_START, COVID_END

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_train_test_split(train, test, dataset_name, test_size):
    """Plot the real training data and test data split."""
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['y'], label='Train', linewidth=1.5)
    plt.plot(test.index, test['y'], label='Test (Actual)', linewidth=1.5)
    plt.title(f'Train vs Test Split for {dataset_name} (Test Size: {test_size} days)')
    plt.xlabel('Date')
    plt.ylabel('Energy Demand (kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save plot
    os.makedirs('output_plots', exist_ok=True)
    plot_path = f'output_plots/train_test_split_{dataset_name}_{test_size}days.png'
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Train/Test split plot saved to {plot_path}")

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
    plot_path = f'output_plots/actual_vs_predict_{model_name}_{dataset_name}_{test_size}days.png'
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
                
            # Define explicit split date explicitly to be passed to train()
            #split_date = df.index[-lag_size_days - 1]
            split_date_str = split_date.strftime('%Y-%m-%d')
            
            # Local copies for plotting and the prediction baseline
            train_df = df[df.index <= split_date].copy()
            test_df = df[df.index > split_date].copy()
            
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
            plot_train_test_split(train_df, test_df, dataset, lag_size_days)
            plot_test_vs_predict(test_df.iloc[:min_len], preds_array[:min_len], dataset, model_name, lag_size_days)
            
    summary_df = pd.DataFrame(results_summary)
    logging.info("\\nFinal Benchmark Summary:\\n" + summary_df.to_string())
    summary_df.to_csv('forecast_metrics_summary.csv', index=False)
    logging.info("Metrics saved to forecast_metrics_summary.csv")

if __name__ == "__main__":

    # fetch and plot
    # li_ds = ['ACN_Caltech', 'ACN_JPL', 'ACN_Office001', 'BeLib', 'AMB_Barcelona']
    datasets = ['ACN_Caltech']
    split_date_str = "2020-06-01"
    prediction_lag_days = [30, 120, 240]
    strategy_params = {
        'epochs': 100, 
        'batch_size': 32,
        'look_back': 14,           # 2 weeks look back
        'prediction_lag_days': prediction_lag_days,
        'learning_rate': 0.001,    # Default used if missing
        'dropout_rate': 0.2,       # Default used if missing
        'activation': 'relu'       # Default used if missing
    }

    run_forecast_pipeline(datasets, strategy_params, split_date_str)
