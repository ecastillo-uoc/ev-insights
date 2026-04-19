import os
from pathlib import Path
import logging
import importlib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine
import psycopg2


# Time series
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# own modules
from src.tfm.tfm_data_fetcher import fetch_dataset_session_details_by_specific_site, fetch_dataset_energy_trends, fetch_dataset_charges_trends, fetch_dataset_plug_ins, fetch_dataset_energy_trends_by_site, fetch_dataset_charges_trends_by_site, fetch_dataset_session_details_by_site, fetch_dataset_session_details_by_station
from src.tfm.tfm_constants import COVID_START, COVID_END

def plot_energy_consumption_trends(dataset: pd.DataFrame, time_col: str = 'timestamp', energy_col: str = 'energy_kwh', 
                                   fig_dir: Path = None, dataset_name: str = None):
    """
    Plots the trends of energy consumption over time.
    
    Args:
        dataset (pd.DataFrame): The input dataset containing EV charging session data.
        time_col (str): The column name denoting the timestamps.
        energy_col (str): The column name denoting the energy provided in kWh.
        save_path (str, optional): The file path to save the generated figure.
    """
    save_path = fig_dir / f'{dataset_name}_energy_trends.png'
    
    # Ensure output directory exists before saving
    save_path.parent.mkdir(parents=True, exist_ok=True)
            
    save_path_str = str(save_path)

    # Optional sorting if not already sorted
    dataset = dataset.sort_values(by=time_col)

    has_station_count = 'station_count' in dataset.columns
    has_session_count = 'session_count' in dataset.columns
    has_duration = 'avg_session_duration_s' in dataset.columns

    n_extra = sum([has_station_count, has_session_count, has_duration])

    if n_extra > 0:
        n_subplots = 1 + n_extra
        height_ratios = [3] + [1] * n_extra
        fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4 + 2 * n_extra), sharex=True,
                                 gridspec_kw={'height_ratios': height_ratios})
        ax1 = axes[0]
        ax_extra = list(axes[1:])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        ax_extra = []

    ax1.plot(dataset[time_col], dataset[energy_col], linestyle='-', linewidth=1, alpha=0.8, color='tab:blue', label='Energy (kWh)')
    ax1.set_title(f"'{dataset_name}' Energy over time")
    ax1.set_ylabel('Energy Provided (kWh)')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left')

    ax_idx = 0
    if has_station_count:
        ax2 = ax_extra[ax_idx]; ax_idx += 1
        ax1_right = ax1.twinx()
        station_ma = dataset['station_count'].rolling(window=7, min_periods=1).mean()
        ax1_right.plot(dataset[time_col], station_ma, linestyle='--', linewidth=1.2, alpha=0.7, color='tab:orange', label='Distinct used Stations (7d MA)')
        ax1_right.set_ylabel('Distinct used Stations (7d MA)', color='tab:orange')
        ax1_right.tick_params(axis='y', labelcolor='tab:orange')
        ax1_right.legend(loc='upper right')

        ax2.bar(dataset[time_col], dataset['station_count'], alpha=0.6, color='tab:orange', width=1.0, label='Distinct daily used Stations')
        ax2.set_ylabel('Distinct Stations')
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.legend(loc='upper left')

    if has_session_count:
        ax3 = ax_extra[ax_idx]; ax_idx += 1
        ax3.bar(dataset[time_col], dataset['session_count'], alpha=0.6, color='tab:green', width=1.0, label='Daily Sessions')
        ax3.set_ylabel('Daily Sessions')
        ax3.grid(True, linestyle='--', alpha=0.4)
        ax3.legend(loc='upper left')

    if has_duration:
        ax4 = ax_extra[ax_idx]; ax_idx += 1
        duration_h = dataset['avg_session_duration_s'] / 3600.0
        p99 = duration_h.quantile(0.99)
        ax4.plot(dataset[time_col], duration_h, linestyle='-', linewidth=1, alpha=0.8, color='tab:purple', label='Daily Avg Session Duration (h)')
        ax4.set_ylabel('Daily Avg Duration (h)')
        ax4.set_ylim(0, p99*1.1)
        ax4.grid(True, linestyle='--', alpha=0.4)
        ax4.legend(loc='upper left')

    if ax_extra:
        ax_extra[-1].set_xlabel('Timestamp')
    else:
        ax1.set_xlabel('Timestamp')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path_str)
        
    plt.show()

def plot_multiple_energy_consumption_trends(datasets: dict, time_col: str = 'timestamp', energy_col: str = 'energy_kwh', 
                                            fig_dir: Path = None, plot_name: str = 'combined_datasets'):
    """
    Plots the trends of energy consumption over time for multiple datasets in a single figure.
    
    Args:
        datasets (dict): Dictionary mapping dataset names to their pandas DataFrames.
        time_col (str): The column name denoting the timestamps.
        energy_col (str): The column name denoting the energy provided in kWh.
        fig_dir (Path, optional): Directory to save the figure.
        plot_name (str): Name to use for the plot title and file name.
    """
    if fig_dir:
        save_path = fig_dir / f'energy_all.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path_str = str(save_path)
    else:
        save_path = None

    plt.figure(figsize=(14, 7))
    
    for name, dataset in datasets.items():
        if not dataset.empty:
            df_sorted = dataset.sort_values(by=time_col)
            # Ensure time_col is datetime for plotting and filling
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
            plt.plot(df_sorted[time_col], df_sorted[energy_col], linestyle='-', linewidth=1, alpha=0.8, label=name)
        
    plt.title(f"{plot_name}")
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Provided (kWh)')
    
    # Highlight the specific gap (COVID-19 alterations)
    gap_start = pd.to_datetime(COVID_START)
    gap_end = pd.to_datetime(COVID_END)
    plt.axvspan(gap_start, gap_end, color='red', alpha=0.2, label='COVID-19 Data Gap')
    
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path_str)
        
    plt.show()




def fetch_and_plot_dataset_trends(dataset_name: str, db_config: dict, fig_dir: Path, site_name: str = None):
    """
    Fetches the dataset by name from the database and plots the energy trends.
    """
    dataset = fetch_dataset_energy_trends(dataset_name, db_config)
    
    if not dataset.empty:
        plot_energy_consumption_trends(dataset, time_col='timestamp', energy_col='energy_kwh', fig_dir=fig_dir, dataset_name=dataset_name)
    else:
        print(f"No data found for the {dataset_name} dataset in the database.")


def plot_energy_consumption_trends_by_site(dataset: pd.DataFrame, time_col: str = 'timestamp', energy_col: str = 'energy_kwh', site_col: str = 'site', fig_dir: Path = None, dataset_name: str = None):
    """
    Plots the trends of energy consumption over time separated by site.
    """
    save_path = fig_dir / f'{dataset_name}_energy_trends_by_site.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path_str = str(save_path)

    plt.figure(figsize=(14, 7))
    dataset[time_col] = pd.to_datetime(dataset[time_col])
    
    sites = sorted(dataset[site_col].dropna().unique())
    for loc in sites:
        loc_data = dataset[dataset[site_col] == loc].sort_values(by=time_col)
        plt.plot(loc_data[time_col], loc_data[energy_col], linestyle='-', linewidth=1.5, alpha=0.8, label=loc)
        
    plt.title(f"'{dataset_name}' Energy over time by site")
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Provided (kWh)')
    plt.legend(title='site', loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path_str)
        
    plt.show()

def fetch_and_plot_energy_trends_by_site(dataset_name: str, db_config: dict, fig_dir: Path):
    """
    Fetches the dataset by site from the database and plots the energy trends.
    """
    dataset = fetch_dataset_energy_trends_by_site(dataset_name, db_config)
    
    if not dataset.empty:
        plot_energy_consumption_trends_by_site(dataset, time_col='timestamp', energy_col='energy_kwh', site_col='site', fig_dir=fig_dir, dataset_name=dataset_name)
    else:
        print(f"No data found for the {dataset_name} dataset in the database.")

def plot_charges_trends(dataset: pd.DataFrame, time_col: str = 'timestamp', count_col: str = 'sessions_count', fig_dir: Path = None, dataset_name: str = None):
    save_path = fig_dir / f'{dataset_name}_sessions_trends.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path_str = str(save_path)

    plt.figure(figsize=(12, 6))
    dataset = dataset.sort_values(by=time_col)
    
    plt.plot(dataset[time_col], dataset[count_col], linestyle='-', linewidth=1, alpha=0.8, color='purple')
    plt.title(f"'{dataset_name}' Number of Daily Sessions over time")
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Sessions')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path_str)
        
    plt.show()

def fetch_and_plot_dataset_charges_trends(dataset_name: str, db_config: dict, fig_dir: Path, site_name: str = None):
    """
    Fetches the dataset by name from the database and plots the charging sessions trends.
    """
    dataset = fetch_dataset_charges_trends(dataset_name, db_config)
    
    if not dataset.empty:
        plot_charges_trends(dataset, time_col='timestamp', count_col='sessions_count', fig_dir=fig_dir, dataset_name=dataset_name)
    else:
        print(f"No data found for the {dataset_name} dataset in the database.")


def plot_charges_trends_by_site(dataset: pd.DataFrame, time_col: str = 'timestamp', count_col: str = 'sessions_count', site_col: str = 'site', fig_dir: Path = None, dataset_name: str = None):
    """
    Plots the trends of charging sessions over time separated by site.
    """
    save_path = fig_dir / f'{dataset_name}_sessions_trends_by_site.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path_str = str(save_path)

    plt.figure(figsize=(14, 7))
    dataset[time_col] = pd.to_datetime(dataset[time_col])
    
    sites = sorted(dataset[site_col].dropna().unique())
    for loc in sites:
        loc_data = dataset[dataset[site_col] == loc].sort_values(by=time_col)
        plt.plot(loc_data[time_col], loc_data[count_col], linestyle='-', linewidth=1.5, alpha=0.8, label=loc)
        
    plt.title(f"'{dataset_name}' Number of Daily Sessions over time by site")
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Sessions')
    plt.legend(title='site', loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path_str)
        
    plt.show()

def fetch_and_plot_charges_trends_by_site(dataset_name: str, db_config: dict, fig_dir: Path):
    """
    Fetches the dataset by site from the database and plots the charging sessions trends.
    """
    dataset = fetch_dataset_charges_trends_by_site(dataset_name, db_config)
    
    if not dataset.empty:
        plot_charges_trends_by_site(dataset, time_col='timestamp', count_col='sessions_count', site_col='site', fig_dir=fig_dir, dataset_name=dataset_name)
    else:
        print(f"No data found for the {dataset_name} dataset in the database.")

def fetch_and_plot_multiple_datasets_trends(dataset_names: list, db_config: dict, fig_dir: Path, plot_name: str = 'combined'):
    """
    Fetches multiple datasets by name from the database and plots their energy trends together.
    """
    datasets = {}
    
    for dataset_name in dataset_names:
        df = fetch_dataset_energy_trends(dataset_name, db_config)
        if not df.empty:
            datasets[dataset_name] = df
        else:
            print(f"No data found for the {dataset_name} dataset in the database.")
            
    if datasets:
        plot_multiple_energy_consumption_trends(datasets, time_col='timestamp', energy_col='energy_kwh', fig_dir=fig_dir, plot_name=plot_name)




def plot_charges_by_weekday(dataset: pd.DataFrame, dataset_name: str, fig_dir: Path):
    """
    Plots the number of charging sessions by weekday.
    """
    save_path = fig_dir / f'{dataset_name}_number_of_charges_by_weekday.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
            
    # Ensure datetime format and extract weekday
    dataset['plug_in_datetime'] = pd.to_datetime(dataset['plug_in_datetime'])
    dataset['plug_in_weekday'] = dataset['plug_in_datetime'].dt.day_name()
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grouped_df = dataset.groupby('plug_in_weekday')['energy_supplied'].count().reset_index()
    grouped_df.rename(columns={'energy_supplied': 'Number of charging sessions'}, inplace=True)
    grouped_df['plug_in_weekday'] = pd.Categorical(grouped_df['plug_in_weekday'], categories=weekday_order, ordered=True)
    grouped_df = grouped_df.sort_values(by='plug_in_weekday')
    
    # Assign different colors for weekdays and weekends
    colors = ['darkblue' if day in ['Saturday', 'Sunday'] else '#1f77b4' for day in grouped_df['plug_in_weekday']]
    
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_df['plug_in_weekday'], grouped_df['Number of charging sessions'], color=colors, edgecolor='black', alpha=0.8)
    plt.title(f"Number of charging sessions by Weekday\nDataset: {dataset_name}", fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Number of Charging Sessions', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(str(save_path))
    plt.show()


def fetch_and_plot_charges_by_weekday(dataset_name: str, db_config: dict, fig_dir: Path, site_name: str = None):
    """
    Fetches the dataset and plots the total number of charging sessions by weekday.
    """
    dataset = fetch_dataset_plug_ins(dataset_name, db_config)
    if not dataset.empty:
        plot_charges_by_weekday(dataset, dataset_name, fig_dir)
    else:
        print(f"No data found for the {dataset_name} dataset in the database.")


def plot_energy_by_weekday(dataset: pd.DataFrame, dataset_name: str, fig_dir: Path):
    """
    Plots the total energy supplied by weekday.
    """
    save_path = fig_dir / f'{dataset_name}_total_energy_by_weekday.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
            
    # Ensure datetime format and extract weekday
    dataset['plug_in_datetime'] = pd.to_datetime(dataset['plug_in_datetime'])
    dataset['plug_in_weekday'] = dataset['plug_in_datetime'].dt.day_name()
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grouped_df = dataset.groupby('plug_in_weekday')['energy_supplied'].sum().reset_index()
    grouped_df.rename(columns={'energy_supplied': 'Total Energy Supplied (kWh)'}, inplace=True)
    grouped_df['plug_in_weekday'] = pd.Categorical(grouped_df['plug_in_weekday'], categories=weekday_order, ordered=True)
    grouped_df = grouped_df.sort_values(by='plug_in_weekday')
    
    # Assign different colors for weekdays and weekends
    colors = ['darkblue' if day in ['Saturday', 'Sunday'] else '#1f77b4' for day in grouped_df['plug_in_weekday']]
    
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_df['plug_in_weekday'], grouped_df['Total Energy Supplied (kWh)'], color=colors, edgecolor='black', alpha=0.8)
    plt.title(f"Total Energy Supplied by Weekday\nDataset: {dataset_name}", fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Total Energy Supplied (kWh)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(str(save_path))
    plt.show()

def fetch_and_plot_energy_by_weekday(dataset_name: str, db_config: dict, fig_dir: Path, site_name: str = None):
    """
    Fetches the dataset and plots the total energy supplied by weekday.
    """
    dataset = fetch_dataset_plug_ins(dataset_name, db_config)
    if not dataset.empty:
        plot_energy_by_weekday(dataset, dataset_name, fig_dir)
    else:
        print(f"No data found for the {dataset_name} dataset in the database.")


def plot_energy_by_month(dataset: pd.DataFrame, dataset_name: str, fig_dir: Path):
    """
    Plots the total energy supplied by month.
    """
    save_path = fig_dir / f'{dataset_name}_total_energy_by_month.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
            
    # Ensure datetime format and extract month
    dataset['plug_in_datetime'] = pd.to_datetime(dataset['plug_in_datetime'])
    dataset['plug_in_month'] = dataset['plug_in_datetime'].dt.month_name()
    
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    grouped_df = dataset.groupby('plug_in_month')['energy_supplied'].sum().reset_index()
    grouped_df.rename(columns={'energy_supplied': 'Total Energy Supplied (kWh)'}, inplace=True)
    grouped_df['plug_in_month'] = pd.Categorical(grouped_df['plug_in_month'], categories=month_order, ordered=True)
    grouped_df = grouped_df.sort_values(by='plug_in_month')
    
    plt.figure(figsize=(12, 6))
    plt.bar(grouped_df['plug_in_month'], grouped_df['Total Energy Supplied (kWh)'], color='#2ca02c', edgecolor='black', alpha=0.8)
    plt.title(f"Total Energy Supplied by Month\nDataset: {dataset_name}", fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Energy Supplied (kWh)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(str(save_path))
    plt.show()

def fetch_and_plot_energy_by_month(dataset_name: str, db_config: dict, fig_dir: Path, site_name: str = None):
    """
    Fetches the dataset and plots the total energy supplied by month.
    """
    dataset = fetch_dataset_plug_ins(dataset_name, db_config)
    if not dataset.empty:
        plot_energy_by_month(dataset, dataset_name, fig_dir)
    else:
        print(f"No data found for the {dataset_name} dataset in the database.")


def plot_charges_by_month(dataset: pd.DataFrame, dataset_name: str, fig_dir: Path):
    """
    Plots the number of charges supplied by month.
    """
    save_path = fig_dir / f'{dataset_name}_total_charges_by_month.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
            
    # Ensure datetime format and extract month
    dataset['plug_in_datetime'] = pd.to_datetime(dataset['plug_in_datetime'])
    dataset['plug_in_month'] = dataset['plug_in_datetime'].dt.month_name()
    
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    grouped_df = dataset.groupby('plug_in_month')['energy_supplied'].count().reset_index()
    grouped_df.rename(columns={'energy_supplied': 'Number of charging sessions'}, inplace=True)
    grouped_df['plug_in_month'] = pd.Categorical(grouped_df['plug_in_month'], categories=month_order, ordered=True)
    grouped_df = grouped_df.sort_values(by='plug_in_month')
    
    plt.figure(figsize=(12, 6))
    plt.bar(grouped_df['plug_in_month'], grouped_df['Number of charging sessions'], color='#9467bd', edgecolor='black', alpha=0.8)
    plt.title(f"Number of Charging Sessions by Month\nDataset: {dataset_name}", fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Charging Sessions', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(str(save_path))
    plt.show()

def fetch_and_plot_charges_by_month(dataset_name: str, db_config: dict, fig_dir: Path, site_name: str = None):
    """
    Fetches the dataset and plots the total number of charges supplied by month.
    """
    dataset = fetch_dataset_plug_ins(dataset_name, db_config)
    if not dataset.empty:
        plot_charges_by_month(dataset, dataset_name, fig_dir)
    else:
        print(f"No data found for the {dataset_name} dataset in the database.")


def tag_covid_period(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Tags the dataset with 'Pre-COVID' and 'Post-COVID' periods based on a threshold.
    Using March 1st, 2020 as the pivot for global COVID-19 behavioral changes.
    """
    dataset['plug_in_datetime'] = pd.to_datetime(dataset['plug_in_datetime'])
    covid_start = pd.to_datetime(COVID_START)
    dataset['covid_period'] = 'Pre-COVID'
    dataset.loc[dataset['plug_in_datetime'] >= covid_start, 'covid_period'] = 'Post-COVID'
    return dataset


def plot_charges_by_weekday_covid(dataset: pd.DataFrame, dataset_name: str, fig_dir: Path):
    """
    Plots the number of charging sessions by weekday, superimposed by Pre and Post COVID.
    Displays both absolute totals and percentages to properly compare structural pattern changes.
    """
    save_path_pct = fig_dir / f'{dataset_name}_charges_weekday_covid_pct.png'
    save_path_abs = fig_dir / f'{dataset_name}_charges_weekday_covid_abs.png'
    save_path_pct.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = tag_covid_period(dataset)
    dataset['plug_in_weekday'] = dataset['plug_in_datetime'].dt.day_name()
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Calculate percentage within each period to show the pattern change
    grouped = dataset.groupby(['covid_period', 'plug_in_weekday'])['energy_supplied'].count().unstack()
    grouped = grouped.reindex(columns=weekday_order)
    
    # Ensure Pre-COVID is plotted before Post-COVID
    if 'Pre-COVID' in grouped.index and 'Post-COVID' in grouped.index:
        grouped = grouped.reindex(['Pre-COVID', 'Post-COVID'])
        
    # 1. Percentage plot
    # Normalize by row sum to get percentage (%) pattern
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    
    ax1 = grouped_pct.T.plot(kind='bar', figsize=(12, 6), alpha=0.8, edgecolor='black', width=0.7)
    
    plt.title(f"Weekly Pattern Change (Pre vs Post COVID) - Sessions (%)\nDataset: {dataset_name}", fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Percentage of Weekly Sessions', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.legend(title='Period')
    plt.tight_layout()
    
    plt.savefig(str(save_path_pct))
    plt.show()

    # 2. Absolute total sessions plot
    ax2 = grouped.T.plot(kind='bar', figsize=(12, 6), alpha=0.8, edgecolor='black', width=0.7)
    
    plt.title(f"Weekly Total Sessions (Pre vs Post COVID) - Absolute Count\nDataset: {dataset_name}", fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Total Sessions Count', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.legend(title='Period')
    plt.tight_layout()
    
    plt.savefig(str(save_path_abs))
    plt.show()


def plot_energy_by_weekday_covid(dataset: pd.DataFrame, dataset_name: str, fig_dir: Path):
    """
    Plots the energy supplied by weekday, superimposed by Pre and Post COVID.
    Displays both absolute totals and percentages.
    """
    save_path_pct = fig_dir / f'{dataset_name}_energy_weekday_covid_pct.png'
    save_path_abs = fig_dir / f'{dataset_name}_energy_weekday_covid_abs.png'
    save_path_pct.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = tag_covid_period(dataset)
    dataset['plug_in_weekday'] = dataset['plug_in_datetime'].dt.day_name()
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    grouped = dataset.groupby(['covid_period', 'plug_in_weekday'])['energy_supplied'].sum().unstack()
    grouped = grouped.reindex(columns=weekday_order)
    
    if 'Pre-COVID' in grouped.index and 'Post-COVID' in grouped.index:
        grouped = grouped.reindex(['Pre-COVID', 'Post-COVID'])
    
    # 1. Percentage plot
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    
    ax1 = grouped_pct.T.plot(kind='bar', figsize=(12, 6), alpha=0.8, edgecolor='black', width=0.7)
    
    plt.title(f"Weekly Pattern Change (Pre vs Post COVID) - Energy (%)\nDataset: {dataset_name}", fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Percentage of Weekly Energy', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.legend(title='Period')
    plt.tight_layout()
    
    plt.savefig(str(save_path_pct))
    plt.show()

    # 2. Absolute total energy plot
    ax2 = grouped.T.plot(kind='bar', figsize=(12, 6), alpha=0.8, edgecolor='black', width=0.7)
    
    plt.title(f"Weekly Total Energy (Pre vs Post COVID) - Absolute (kWh)\nDataset: {dataset_name}", fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Total Energy (kWh)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.legend(title='Period')
    plt.tight_layout()
    
    plt.savefig(str(save_path_abs))
    plt.show()


def fetch_and_plot_covid_patterns(dataset_name: str, db_config: dict, fig_dir: Path):
    """
    Fetches the dataset and plots the patterns split by COVID.
    """
    dataset = fetch_dataset_plug_ins(dataset_name, db_config)
    # Filter datasets that have data across both periods
    if not dataset.empty:
        plot_charges_by_weekday_covid(dataset, dataset_name, fig_dir)
        plot_energy_by_weekday_covid(dataset, dataset_name, fig_dir)
    else:
        print(f"No data found for the {dataset_name} dataset in the database.")

def fetch_and_plot_time_serie_decomposition(dataset_name: str, db_config: dict, fig_dir: Path, site_name: str = None):
    
    # 1. Fetch data already grouped by day
    dataset = fetch_dataset_energy_trends(dataset_name, db_config)
    
    if dataset.empty:
        print(f"No data for {dataset_name}")
        return

    # 2. Set the datetime index and sort
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset.set_index('timestamp', inplace=True)
    dataset.sort_index(inplace=True)

    # 3. Run decomposition on the specific target column with a meaningful period
    time_serie_decomposition = sm.tsa.seasonal_decompose(
        x=dataset['energy_kwh'].dropna(),
        model='additive',
        period=7  # 7 days for a weekly pattern
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    axes[0].plot(time_serie_decomposition.observed, label='Observed', color='blue')
    axes[0].set_ylabel('Observed')
    axes[0].legend(loc='upper left')
    axes[0].set_title(f'Time Series Decomposition (Energy) - {dataset_name}')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    axes[1].plot(time_serie_decomposition.trend, label='Trend', color='orange')
    axes[1].set_ylabel('Trend')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    axes[2].plot(time_serie_decomposition.seasonal, label='Seasonal', color='green')
    axes[2].set_ylabel('Seasonal')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    axes[3].scatter(time_serie_decomposition.resid.index, time_serie_decomposition.resid, label='Residual', color='red', alpha=0.5, s=10)
    axes[3].plot(time_serie_decomposition.resid, color='red', alpha=0.3)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')
    axes[3].legend(loc='upper left')
    axes[3].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = fig_dir / f'{dataset_name}_ts_decomposition.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    
    plt.show()

def fetch_and_plot_charges_time_serie_decomposition(dataset_name: str, db_config: dict, fig_dir: Path, site_name: str = None):
    
    # 1. Fetch data already grouped by day
    dataset = fetch_dataset_charges_trends(dataset_name, db_config)
    
    if dataset.empty:
        print(f"No data for {dataset_name}")
        return

    # 2. Set the datetime index and sort
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset.set_index('timestamp', inplace=True)
    dataset.sort_index(inplace=True)

    # 3. Run decomposition on the specific target column with a meaningful period
    time_serie_decomposition = sm.tsa.seasonal_decompose(
        x=dataset['sessions_count'].dropna(),
        model='additive',
        period=7  # 7 days for a weekly pattern
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    axes[0].plot(time_serie_decomposition.observed, label='Observed', color='purple')
    axes[0].set_ylabel('Observed')
    axes[0].legend(loc='upper left')
    axes[0].set_title(f'Time Series Decomposition (Sessions) - {dataset_name}')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    axes[1].plot(time_serie_decomposition.trend, label='Trend', color='orange')
    axes[1].set_ylabel('Trend')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    axes[2].plot(time_serie_decomposition.seasonal, label='Seasonal', color='green')
    axes[2].set_ylabel('Seasonal')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    axes[3].scatter(time_serie_decomposition.resid.index, time_serie_decomposition.resid, label='Residual', color='red', alpha=0.5, s=10)
    axes[3].plot(time_serie_decomposition.resid, color='red', alpha=0.3)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')
    axes[3].legend(loc='upper left')
    axes[3].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = fig_dir / f'{dataset_name}_ts_decomposition_sessions.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    
    plt.show()

def plot_session_boxplots_by_site(dataset: pd.DataFrame, dataset_name: str, fig_dir: Path):
    """
    Plots boxplots of session energy and duration by site.
    """
    # Calculate duration in hours
    dataset['plug_in_datetime'] = pd.to_datetime(dataset['plug_in_datetime'])
    dataset['plug_out_datetime'] = pd.to_datetime(dataset['plug_out_datetime'])
    dataset['duration_hours'] = (dataset['plug_out_datetime'] - dataset['plug_in_datetime']).dt.total_seconds() / 3600.0
    
    # Filter reasonable durations (e.g. drop negative or wildly huge durations > 7 days) if needed
    dataset = dataset[(dataset['duration_hours'] >= 0) & (dataset['duration_hours'] < 24 * 7)]
    dataset = dataset[dataset['energy_supplied'] >= 0]
    
    # Sort sites alphabetically for consistency
    dataset = dataset.dropna(subset=['site'])
    dataset['site'] = pd.Categorical(dataset['site'], categories=sorted(dataset['site'].unique()), ordered=True)

  

    # 1. Energy Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=dataset, x='site', y='energy_supplied', order=sorted(dataset['site'].unique()), palette='husl')
    plt.title(f'Session Energy Distribution by Site\nDataset: {dataset_name}', fontsize=14)
    plt.ylabel('Energy Supplied (kWh)', fontsize=12)
    plt.xlabel('Site', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path_energy = fig_dir / f'{dataset_name}_boxplot_energy_by_site.png'
    save_path_energy.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path_energy))
    plt.show()

    # 2. Duration Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=dataset, x='site', y='duration_hours', order=sorted(dataset['site'].unique()), palette='husl')
    plt.title(f'Session Duration Distribution by Site\nDataset: {dataset_name}', fontsize=14)
    plt.ylabel('Duration (Hours)', fontsize=12)
    plt.xlabel('Site', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path_duration = fig_dir / f'{dataset_name}_boxplot_duration_by_site.png'
    save_path_duration.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path_duration))
    plt.show()

def fetch_and_plot_session_boxplots_by_site(dataset_name: str, db_config: dict, fig_dir: Path):
    """
    Fetches the session details by dataset (and groups by site internally) and generates the energy and duration boxplots.
    """
    dataset = fetch_dataset_session_details_by_site, fetch_dataset_session_details_by_station, fetch_dataset_session_details_by_station(dataset_name, db_config)
    if not dataset.empty:
        plot_session_boxplots_by_site(dataset, dataset_name, fig_dir)
    else:
        print(f"No session details found for the {dataset_name} dataset in the database.")

def fetch_and_plot_session_boxplots_by_specific_site(dataset_name: str, site_name: str, db_config: dict, fig_dir: Path):
    """
    Fetches the session details by a specific dataset and site then generates the energy and duration boxplots.
    """
    dataset = fetch_dataset_session_details_by_specific_site(dataset_name, site_name, db_config)
    if not dataset.empty:
        plot_session_boxplots_by_site(dataset, f"{dataset_name} ({site_name})", fig_dir)
    else:
        print(f"No session details found for the {dataset_name} dataset at site {site_name} in the database.")

def fetch_and_plot_session_boxplots_by_station(station_ids: tuple = None, city: str = None, db_config: dict = None, fig_dir: Path = None):
    """
    Fetches the session details by explicit station IDs and/or city, and generates the energy and duration boxplots.
    """
    dataset = fetch_dataset_session_details_by_station(station_ids=station_ids, city=city, db_config=db_config)
    if not dataset.empty:
        dataset['plug_in_datetime'] = pd.to_datetime(dataset['plug_in_datetime'])
        dataset['plug_out_datetime'] = pd.to_datetime(dataset['plug_out_datetime'])
        dataset['duration_hours'] = (dataset['plug_out_datetime'] - dataset['plug_in_datetime']).dt.total_seconds() / 3600.0
        
        title_parts = []
        if station_ids:
            title_parts.append(f"Stations_{'_'.join(map(str, station_ids))}")
        if city:
            title_parts.append(f"City_{city}")
            
        title = "_".join(title_parts) if title_parts else "All_Stations"
        
        plot_session_boxplots_by_site(dataset, title, fig_dir)
    else:
        print(f"No session details found for the station IDs {station_ids} and city {city}.")


def main():
    """
    Main execution function.
    """
    FIG_DIR = Path('/home/ecastillo/dev/M2_882_TFM/tfm/doc/figures/')

    # Database connection parameters (from project environment)
    db_config = {
        'dbname': 'evinsights',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': 5432
    }
    
    # Different plots needed
    
    # Example plot with multiple datasets
    # fetch_and_plot_multiple_datasets_trends(['ACN_Caltech', 'ACN_JPL', 'ACN_Office001'], db_config, FIG_DIR, plot_name='COVID-19 data gap and pattern disturbance')

    # fetch and plot
    # li_ds = ['ACN_Caltech', 'ACN_JPL', 'ACN_Office001', 'BeLib', 'AMB_Barcelona']
    # li_ds = [('Norway_12loc', 'OSL_T'), ('Norway_12loc', 'OSL_S'), 'ACN_Caltech']
    li_ds = ['Dundee']
    for item in li_ds:
        if isinstance(item, tuple):
            ds, site = item
        else:
            ds = item
            site = None

        print(f"\n--- Processing dataset: {ds}" + (f", site: {site}" if site else "") + " ---")

        # Energy
        fetch_and_plot_dataset_trends(ds, db_config, FIG_DIR, site_name=site)
        # fetch_and_plot_energy_by_weekday(ds, db_config, FIG_DIR, site_name=site)
        # fetch_and_plot_energy_by_month(ds, db_config, FIG_DIR, site_name=site)
        fetch_and_plot_time_serie_decomposition(ds, db_config, FIG_DIR, site_name=site)
        
        # Sessions / Charges
        # fetch_and_plot_dataset_charges_trends(ds, db_config, FIG_DIR, site_name=site)
        # fetch_and_plot_charges_by_weekday(ds, db_config, FIG_DIR, site_name=site)
        # fetch_and_plot_charges_by_month(ds, db_config, FIG_DIR, site_name=site)
        # fetch_and_plot_charges_time_serie_decomposition(ds, db_config, FIG_DIR, site_name=site)
        
        # Boxplots
        if site:
            fetch_and_plot_session_boxplots_by_specific_site(ds, site, db_config, FIG_DIR)
        else:
            pass
            #fetch_and_plot_session_boxplots_by_site(ds, db_config, FIG_DIR)
            # Also if no site is provided, we might want to plot the overall grouping by site
            #fetch_and_plot_energy_trends_by_site(ds, db_config, FIG_DIR)
            #fetch_and_plot_charges_trends_by_site(ds, db_config, FIG_DIR)

    # fetch_and_plot_session_boxplots_by_station(city='Sant Cugat del Vall?s', db_config=db_config, fig_dir=FIG_DIR)
    # fetch_and_plot_session_boxplots_by_station(station_ids=(141, 142), db_config=db_config, fig_dir=FIG_DIR)


if __name__ == '__main__':
    main()

