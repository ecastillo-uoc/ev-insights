import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


logging.basicConfig()
_logger = logging.getLogger("analysis.predictions")
_logger.setLevel(logging.DEBUG)


def plot_predictions(actual_dates, actual_values, future_dates, 
                     predictions, 
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
    plt.plot(future_dates, predictions, label="LSTM Predictions", color='orange', linestyle='--', linewidth=2)
    
    # Formatting
    plt.title(f"{title} - {prediction_days} Days Horizon")
    plt.xlabel("Date")
    plt.ylabel("Energy")
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