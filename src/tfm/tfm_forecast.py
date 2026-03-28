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

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_forecast(train, test, predictions, dataset_name, model_name, test_size):
    """Plot the real training data, test data, and predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['y'], label='Train')
    plt.plot(test.index, test['y'], label='Test (Actual)')
    plt.plot(test.index, predictions, label='Predictions', linestyle='--')
    plt.title(f'LSTM Forecast for {dataset_name} ({test_size} days)')
    plt.xlabel('Date')
    plt.ylabel('Energy Demand (kWh)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('output_plots', exist_ok=True)
    plot_path = f'output_plots/forecast_{model_name}_{dataset_name}_{test_size}days.png'
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Plot saved to {plot_path}")

def run_forecast_pipeline(datasets, prediction_lag_days):
    
    models_dir = 'output_models'
    os.makedirs(models_dir, exist_ok=True)

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
                
            train_df = df.iloc[:-lag_size_days]
            test_df = df.iloc[-lag_size_days:]
            
            # 2. Train model using LSTMModelStrategy
            strategy = LSTMModelStrategy()
            model_name_prefix = f"lstm_{lag_size_days}d"
            
            # Format train df 
            train_df = train_df.copy()
            train_df['dataset_name'] = dataset
            
            strategy_params = {
                'epochs': 20, 
                'batch_size': 32,
                'look_back': 14, # 2 weeks look back
                'prediction_days': lag_size_days
            }
            train_df.attrs['lstm_params'] = strategy_params
            
            logging.info("Training model...")
            trained_models_dict = strategy.train(df=train_df, feature_columns=[], target_column='y', dataset_names=[dataset], model_name_prefix=model_name_prefix)
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
            models_name = "lstm"
            plot_forecast(train_df, test_df.iloc[:min_len], preds_array[:min_len], dataset, model_name, lag_size_days)
            
    summary_df = pd.DataFrame(results_summary)
    logging.info("\\nFinal Benchmark Summary:\\n" + summary_df.to_string())
    summary_df.to_csv('forecast_metrics_summary.csv', index=False)
    logging.info("Metrics saved to forecast_metrics_summary.csv")

if __name__ == "__main__":

    # fetch and plot
    # li_ds = ['ACN_Caltech', 'ACN_JPL', 'ACN_Office001', 'BeLib', 'AMB_Barcelona']
    datasets = ['ACN_Caltech']
    prediction_lag_days = [30, 120, 240]
    run_forecast_pipeline(datasets, prediction_lag_days)
