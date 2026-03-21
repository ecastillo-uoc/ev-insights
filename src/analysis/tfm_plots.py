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

def fetch_and_plot_dataset_trends(dataset_name: str, db_config: dict, fig_dir: Path):
    """
    Fetches the dataset by name from the database and plots the energy trends.
    """
    save_path = fig_dir / f'{dataset_name}_energy_trends.png'
    
    query = """
    SELECT 
        cs.plug_out_datetime as timestamp, 
        cs.energy_supplied as energy_kwh
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
        
        if not dataset.empty:
            # Ensure output directory exists before saving
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            plot_energy_consumption_trends(dataset, time_col='timestamp', energy_col='energy_kwh', save_path=str(save_path))
        else:
            print(f"No data found for the {dataset_name} dataset in the database.")
            
    except psycopg2.OperationalError as e:
        print(f"Failed to connect to the database: {e}")
    except Exception as e:
        print(f"An error occurred: Query: {query}\n -- {e}")

    pass


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
    fetch_and_plot_dataset_trends('ACN_Caltech', db_config, FIG_DIR)

if __name__ == '__main__':
    main()

