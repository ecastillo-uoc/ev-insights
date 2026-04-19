"""
Forecast-specific matplotlib plots.

These helpers produce the train/test-split and actual-vs-predicted
visualisations saved to ``output_plots/``.  They are called by the
forecast pipeline (``src.forecast.pipeline``) after each horizon
evaluation.
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_train_test_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_name: str,
    forecast_horizon: int,
    zoom_range=None,
    model_name: str | None = None,
    warm_up_days: int = 0,
) -> str:
    """Plot the target variable with distinct colours for train and test periods.

    Returns the path of the saved full-range plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['y'], label='Train', linewidth=1.5, color='#1f77b4')
    plt.plot(test.index, test['y'], label='Test', linewidth=1.5, color='#ff7f0e')

    split_date = train.index.max()
    plt.axvline(x=split_date, color='grey', linestyle='--', linewidth=1, label='Train / Test split')

    if warm_up_days > 0 and warm_up_days < len(test):
        eval_start = test.index[warm_up_days]
        plt.axvspan(test.index[0], eval_start, alpha=0.12, color='orange', label='Warm-up (look_back)')
        plt.axvline(x=eval_start, color='orange', linestyle=':', linewidth=1, label='Eval start')

    title = f'Train/Test Split for {dataset_name}'
    if model_name:
        title += f'\nModel: {model_name}'
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Energy delivered (kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs('output_plots', exist_ok=True)
    suffix = f'_{model_name}' if model_name else ''
    plot_path = f'output_plots/{dataset_name}_train_test_split{suffix}_{forecast_horizon}days.png'
    plt.savefig(plot_path)
    logger.info("Train/Test split plot saved to %s", plot_path)

    if zoom_range:
        plt.xlim(pd.to_datetime(zoom_range[0]), pd.to_datetime(zoom_range[1]))
        plot_path_zoom = f'output_plots/{dataset_name}_train_test_split{suffix}_{forecast_horizon}days_zoom.png'
        plt.savefig(plot_path_zoom)
        logger.info("Train/Test split zoom plot saved to %s", plot_path_zoom)

    plt.close()
    return plot_path


def plot_test_vs_predict(
    test: pd.DataFrame,
    predictions,
    dataset_name: str,
    model_name: str,
    forecast_horizon: int,
) -> None:
    """Plot test data (actual) against predictions and save to disk."""
    min_len = min(len(test), len(predictions))
    if min_len < len(test) or min_len < len(predictions):
        logger.warning(
            "Length mismatch: test=%d, predictions=%d. Truncating to %d.",
            len(test), len(predictions), min_len,
        )
    test = test.iloc[:min_len]
    predictions = predictions[:min_len]
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['y'], label='Test (Actual)', linewidth=1.5, color='green')
    plt.plot(test.index, predictions, label='Predictions', linestyle='--', linewidth=1.5, color='red')
    plt.title(f'Actual vs Predictions for {dataset_name} ({forecast_horizon} days)\nModel: {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Energy Demand (kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs('output_plots', exist_ok=True)
    plot_path = f'output_plots/{dataset_name}_actual_vs_predict_{model_name}_{forecast_horizon}days.png'
    plt.savefig(plot_path)
    plt.close()
    logger.info("Actual vs Predict plot saved to %s", plot_path)
