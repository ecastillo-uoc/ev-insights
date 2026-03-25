import os
import logging
import importlib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import psycopg2

def plot_energy_consumption_trends(dataset: pd.DataFrame, time_col: str = 'timestamp', energy_col: str = 'energy_kwh', fig_dir: Path = None, dataset_name: str = None):
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

    plt.figure(figsize=(12, 6))
    
    # Optional sorting if not already sorted
    dataset = dataset.sort_values(by=time_col)
    
    plt.plot(dataset[time_col], dataset[energy_col], linestyle='-', linewidth=1, alpha=0.8)
    plt.title(f"'{dataset_name}' Energy over time ")
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Provided (kWh)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path_str)
        
    plt.show()

def plot_multiple_energy_consumption_trends(datasets: dict, time_col: str = 'timestamp', energy_col: str = 'energy_kwh', fig_dir: Path = None, plot_name: str = 'combined_datasets'):
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
    gap_start = pd.to_datetime('2020-08-01')
    gap_end = pd.to_datetime('2020-11-20')
    plt.axvspan(gap_start, gap_end, color='red', alpha=0.2, label='COVID-19 Data Gap')
    
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path_str)
        
    plt.show()

def fetch_dataset_energy_trends(dataset_name: str, db_config: dict) -> pd.DataFrame:
    """
    Fetches the dataset by name from the database and returns the energy trends as a DataFrame.
    """
    query = """
    SELECT 
        DATE(cs.plug_out_datetime) as timestamp, 
        SUM(cs.energy_supplied) as energy_kwh
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d 
        ON cs.fk_dataset_id = d.id
    WHERE d.name = %(dataset_name)s
    GROUP BY DATE(cs.plug_out_datetime)
    ORDER BY timestamp
    """
    
    try:       
        # Create SQLAlchemy engine URL: postgresql://user:password@host:port/dbname
        engine_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        engine = create_engine(engine_url)
        
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        
        return dataset
            
    except Exception as e:
        print(f"An error occurred while fetching {dataset_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()


def fetch_and_plot_dataset_trends(dataset_name: str, db_config: dict, fig_dir: Path):
    """
    Fetches the dataset by name from the database and plots the energy trends.
    """
    dataset = fetch_dataset_energy_trends(dataset_name, db_config)
    
    if not dataset.empty:
        plot_energy_consumption_trends(dataset, time_col='timestamp', energy_col='energy_kwh', fig_dir=fig_dir, dataset_name=dataset_name)
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


def fetch_dataset_plug_ins(dataset_name: str, db_config: dict) -> pd.DataFrame:
    """
    Fetches the dataset by name from the database and returns the plug_in_datetime and energy_supplied.
    """
    query = """
    SELECT 
        cs.plug_in_datetime, 
        cs.energy_supplied
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d 
        ON cs.fk_dataset_id = d.id
    WHERE d.name = %(dataset_name)s
    """
    
    try:
        from sqlalchemy import create_engine
        
        # Create SQLAlchemy engine URL: postgresql://user:password@host:port/dbname
        engine_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        engine = create_engine(engine_url)
        
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        
        return dataset
            
    except Exception as e:
        print(f"An error occurred while fetching plug_in data for {dataset_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()


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


def fetch_and_plot_charges_by_weekday(dataset_name: str, db_config: dict, fig_dir: Path):
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

def fetch_and_plot_energy_by_weekday(dataset_name: str, db_config: dict, fig_dir: Path):
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

def fetch_and_plot_energy_by_month(dataset_name: str, db_config: dict, fig_dir: Path):
    """
    Fetches the dataset and plots the total energy supplied by month.
    """
    dataset = fetch_dataset_plug_ins(dataset_name, db_config)
    if not dataset.empty:
        plot_energy_by_month(dataset, dataset_name, fig_dir)
    else:
        print(f"No data found for the {dataset_name} dataset in the database.")


def tag_covid_period(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Tags the dataset with 'Pre-COVID' and 'Post-COVID' periods based on a threshold.
    Using March 1st, 2020 as the pivot for global COVID-19 behavioral changes.
    """
    dataset['plug_in_datetime'] = pd.to_datetime(dataset['plug_in_datetime'])
    covid_start = pd.to_datetime('2020-03-01')
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
    # fetch_and_plot_dataset_trends('ACN_Caltech', db_config, FIG_DIR)
    #fetch_and_plot_dataset_trends('ACN_JPL', db_config, FIG_DIR)
    #fetch_and_plot_dataset_trends('ACN_Office001', db_config, FIG_DIR)
    #fetch_and_plot_dataset_trends('BeLib', db_config, FIG_DIR)
    #fetch_and_plot_dataset_trends('AMB_Barcelona', db_config, FIG_DIR)
    
    # Example plot with multiple datasets
    # fetch_and_plot_multiple_datasets_trends(['ACN_Caltech', 'ACN_JPL', 'ACN_Office001'], db_config, FIG_DIR, plot_name='COVID-19 data gap and pattern disturbance')

    # fetch and plot weekday
    li_ds = ['ACN_Caltech', 'ACN_JPL', 'ACN_Office001', 'BeLib', 'AMB_Barcelona']
    for ds in li_ds:
        # fetch_and_plot_charges_by_weekday(ds, db_config, FIG_DIR)
        # fetch_and_plot_energy_by_weekday(ds, db_config, FIG_DIR)
        # fetch_and_plot_energy_by_month(ds, db_config, FIG_DIR)
        fetch_and_plot_covid_patterns(ds, db_config, FIG_DIR)

if __name__ == '__main__':
    main()

