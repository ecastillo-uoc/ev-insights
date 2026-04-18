import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


logging.basicConfig()
_logger = logging.getLogger("analysis.predictions")
_logger.setLevel(logging.DEBUG)


def plot_predictions_energy(actual_dates, actual_values, future_dates, 
                     prediced_values, 
                     title:str,
                     prediction_days:str, 
                     save_dir=None, show_images:bool=False):
    """
    Plots the real values and the predicted values.
    
    Args:
        actual_dates (array-like): Dates for the historical actual data.
        actual_values (array-like): The historical real data values.
        future_dates (array-like): The dates corresponding to the future predictions.
        predictions (array-like): Inverse-scaled predictions to plot against actuals.
        title (str): Name of the plot
        prediction_days (str): The prediction horizon in days (e.g., "30 Days").
        save_dir (Path or str): Directory to save the plot. Optional.
    """
    
    plt.figure(figsize=(14, 7))
    
    # Plot real data
    plt.plot(actual_dates, actual_values, label="Actual Data", color='blue', linewidth=1.5)
    
    # Plot predicted data
    plt.plot(future_dates, prediced_values, label="LSTM Predictions", color='orange', linestyle='--', linewidth=2)
    
    # Formatting
    plt.title(f"Forecast: model:{title} - {prediction_days} Days Horizon")
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    if save_dir:
        plot_fn = f"{title}_{prediction_days}.png".replace(" ","_")
        save_path = Path(save_dir) / plot_fn
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path))
        _logger.info(f"Plot saved to: {save_path}")
        
    if show_images:
        plt.show()
    else:
        plt.close()


def plot_forecast_results(predict_results, forecast_name="Forecast",
                          ylabel="Value", save_dir=None):
    """
    Build a matplotlib Figure from a predict output dict.

    Handles three layouts:
      1. Actual vs Predicted (dates + values + actuals)
      2. Future forecast only  (dates + values, no actuals)
      3. Values only fallback  (values without dates → integer x-axis)

    Args:
        predict_results: dict with keys from the predict output,
                         e.g. {'values': [...], 'dates': [...], 'actuals': [...], ...}
        forecast_name:   label for the plot title
        ylabel:          y-axis label
        save_dir:        optional directory to save the plot as PNG

    Returns:
        matplotlib.figure.Figure  (caller can pass to st.pyplot)
    """
    pred_data = predict_results.get('predict', predict_results)

    values = pred_data.get('values')
    dates = pred_data.get('dates')
    actuals = pred_data.get('actuals')

    if values is None:
        _logger.warning("No 'values' key in predict results – nothing to plot")
        return None

    values = np.asarray(values, dtype=float)

    # Parse dates if present
    if dates is not None:
        dates = pd.to_datetime(dates)

    fig, ax = plt.subplots(figsize=(14, 6))

    has_actuals = actuals is not None and len(actuals) > 0

    if has_actuals:
        actuals = np.asarray(actuals, dtype=float)
        min_len = min(len(actuals), len(values))
        actuals, values = actuals[:min_len], values[:min_len]
        if dates is not None:
            dates = dates[:min_len]
        x_axis = dates if dates is not None else np.arange(min_len)
        ax.plot(x_axis, actuals, label="Actual", color='steelblue', linewidth=1.2)
        ax.plot(x_axis, values, label="Predicted", color='orange',
                linestyle='--', linewidth=1.5, alpha=0.85)
        ax.set_title(f"{forecast_name} — Actual vs Predicted")
    elif dates is not None:
        # Future forecast (e.g. LSTM schedule)
        ax.plot(dates, values, label="Predicted", color='orange',
                linestyle='-', linewidth=2, marker='o', markersize=3)
        ax.set_title(f"{forecast_name} — Forecast")
    else:
        ax.plot(values, label="Predicted", color='orange', linewidth=1.5)
        ax.set_title(f"{forecast_name} — Predicted Values")

    ax.set_xlabel("Date" if dates is not None else "Index")
    ax.set_ylabel(ylabel)

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()

    if save_dir:
        plot_fn = f"{forecast_name.replace(' ', '_')}_prediction.png"
        save_path = Path(save_dir) / plot_fn
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path))
        _logger.info(f"Plot saved to: {save_path}")

    return fig