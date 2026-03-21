import os
import logging
import importlib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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

def fetch_and_plot_dataset_trends(dataset_name: str, db_config: dict, fig_dir: Path):
    """
    Fetches the dataset by name from the database and plots the energy trends.
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
        from sqlalchemy import create_engine
        
        # Create SQLAlchemy engine URL: postgresql://user:password@host:port/dbname
        engine_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        engine = create_engine(engine_url)
        
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        
        if not dataset.empty:
            plot_energy_consumption_trends(dataset, time_col='timestamp', energy_col='energy_kwh', fig_dir=fig_dir, dataset_name=dataset_name)
        else:
            print(f"No data found for the {dataset_name} dataset in the database.")
            
    except psycopg2.OperationalError as e:
        print(f"Failed to connect to the database: {e}")
    except Exception as e:
        print(f"An error occurred: Query: {query}\n -- {e}")


def fetch_and_plot_multiple_datasets_trends(dataset_names: list, db_config: dict, fig_dir: Path, plot_name: str = 'combined'):
    """
    Fetches multiple datasets by name from the database and plots their energy trends together.
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
    
    datasets = {}
    
    try:
        from sqlalchemy import create_engine
        
        # Create SQLAlchemy engine URL
        engine_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        engine = create_engine(engine_url)
        
        with engine.connect() as conn:
            for dataset_name in dataset_names:
                df = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
                if not df.empty:
                    datasets[dataset_name] = df
                else:
                    print(f"No data found for the {dataset_name} dataset in the database.")
                    
        if datasets:
            plot_multiple_energy_consumption_trends(datasets, time_col='timestamp', energy_col='energy_kwh', fig_dir=fig_dir, plot_name=plot_name)
            
    except Exception as e:
        print(f"An error occurred: {e}")


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
    
    # You can reuse this function for any other datasets
    # fetch_and_plot_dataset_trends('ACN_Caltech', db_config, FIG_DIR)
    #fetch_and_plot_dataset_trends('ACN_JPL', db_config, FIG_DIR)
    #fetch_and_plot_dataset_trends('ACN_Office001', db_config, FIG_DIR)
    #fetch_and_plot_dataset_trends('BeLib', db_config, FIG_DIR)
    #fetch_and_plot_dataset_trends('AMB_Barcelona', db_config, FIG_DIR)
    
    # Example plot with multiple datasets
    fetch_and_plot_multiple_datasets_trends(['ACN_Caltech', 'ACN_JPL', 'ACN_Office001'], db_config, FIG_DIR, plot_name='COVID-19 data gap and pattern disturbance')

if __name__ == '__main__':
    main()

