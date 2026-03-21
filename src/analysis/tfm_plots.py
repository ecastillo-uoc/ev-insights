import os
import logging
import importlib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def plot_energy_consumption_trends(dataset: pd.DataFrame, time_col: str = 'timestamp', energy_col: str = 'energy_kwh', save_path: str = None):
    """
    Plots the trends of energy consumption over time.
    
    Args:
        dataset (pd.DataFrame): The input dataset containing EV charging session data.
        time_col (str): The column name denoting the timestamps.
        energy_col (str): The column name denoting the energy provided in kWh.
        save_path (str, optional): The file path to save the generated figure.
    """
    plt.figure(figsize=(12, 6))
    
    # Optional sorting if not already sorted
    dataset = dataset.sort_values(by=time_col)
    
    plt.plot(dataset[time_col], dataset[energy_col], marker='.', linestyle='-', linewidth=1, alpha=0.8)
    plt.title('Energy Consumption Trends Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Provided (kWh)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
