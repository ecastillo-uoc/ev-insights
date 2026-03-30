import pandas as pd
from sqlalchemy import create_engine

DEFAULT_DB_CONFIG = {
    'dbname': 'evinsights',
    'user': 'evinsights',
    'password': 'evinsights',
    'host': 'localhost',
    'port': 5432
}

def get_engine(db_config=None):
    """Creates a SQLAlchemy engine from the configuration."""
    if db_config is None:
        db_config = DEFAULT_DB_CONFIG
        
    engine_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    return create_engine(engine_url)

def fetch_dataset_energy_trends(dataset_name: str, db_config: dict = None) -> pd.DataFrame:
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
        engine = get_engine(db_config)
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        return dataset
    except Exception as e:
        print(f"An error occurred while fetching {dataset_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()

def fetch_dataset_charges_trends(dataset_name: str, db_config: dict = None) -> pd.DataFrame:
    """
    Fetches the dataset by name from the database and returns the daily session counts as a DataFrame.
    """
    query = """
    SELECT 
        DATE(cs.plug_in_datetime) as timestamp, 
        COUNT(cs.id) as sessions_count
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d 
        ON cs.fk_dataset_id = d.id
    WHERE d.name = %(dataset_name)s
    GROUP BY DATE(cs.plug_in_datetime)
    ORDER BY timestamp
    """
    try:       
        engine = get_engine(db_config)
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        return dataset
    except Exception as e:
        print(f"An error occurred while fetching {dataset_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()

def fetch_dataset_plug_ins(dataset_name: str, db_config: dict = None) -> pd.DataFrame:
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
        engine = get_engine(db_config)
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        return dataset
    except Exception as e:
        print(f"An error occurred while fetching plug_in data for {dataset_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()

def fetch_daily_energy_for_forecast(dataset_name: str, db_config: dict = None) -> pd.DataFrame:
    """
    Fetches daily energy demand data and formats it for forecasting models (Prophet/LSTM).
    Fills missing days with 0. 
    Returns DataFrame with a DatetimeIndex 'ds' and values 'y'.
    """
    df = fetch_dataset_energy_trends(dataset_name, db_config)
    if df.empty:
        return df
        
    df.rename(columns={'timestamp': 'ds', 'energy_kwh': 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    
    # Ensure regular daily frequency, filling missing days with 0
    df = df.asfreq('D', fill_value=0)
    
    return df

def fetch_dataset_energy_trends_by_location(dataset_name: str, db_config: dict = None) -> pd.DataFrame:
    """
    Fetches the dataset by name from the database and returns the energy trends, grouped by location (station prefix).
    """
    query = """
    SELECT 
        DATE(cs.plug_out_datetime) as timestamp, 
        SUBSTRING(st.orig_id FROM '^(OSL_1|OSL_2|OSL_S|OSL_T|BAR_2|TRO_R|ASK|BAR|BER|BOD|KRO|TRO)') AS location,
        SUM(cs.energy_supplied) as energy_kwh
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d 
        ON cs.fk_dataset_id = d.id
    JOIN evinsights."ChargingStation" st
        ON cs.fk_charging_station_id = st.id
    WHERE d.name = %(dataset_name)s
    GROUP BY DATE(cs.plug_out_datetime), SUBSTRING(st.orig_id FROM '^(OSL_1|OSL_2|OSL_S|OSL_T|BAR_2|TRO_R|ASK|BAR|BER|BOD|KRO|TRO)')
    ORDER BY timestamp
    """
    try:       
        engine = get_engine(db_config)
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        return dataset
    except Exception as e:
        print(f"An error occurred while fetching by location for {dataset_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()

def fetch_dataset_charges_trends_by_location(dataset_name: str, db_config: dict = None) -> pd.DataFrame:
    """
    Fetches the dataset by name from the database and returns the daily session 
    counts grouped by location (station prefix).
    """
    query = """
    SELECT 
        DATE(cs.plug_in_datetime) as timestamp, 
        SUBSTRING(st.orig_id FROM '^(OSL_1|OSL_2|OSL_S|OSL_T|BAR_2|TRO_R|ASK|BAR|BER|BOD|KRO|TRO)') AS location,
        COUNT(cs.id) as sessions_count
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d 
        ON cs.fk_dataset_id = d.id
    JOIN evinsights."ChargingStation" st
        ON cs.fk_charging_station_id = st.id
    WHERE d.name = %(dataset_name)s
    GROUP BY DATE(cs.plug_in_datetime), SUBSTRING(st.orig_id FROM '^(OSL_1|OSL_2|OSL_S|OSL_T|BAR_2|TRO_R|ASK|BAR|BER|BOD|KRO|TRO)')
    ORDER BY timestamp
    """
    try:       
        engine = get_engine(db_config)
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        return dataset
    except Exception as e:
        print(f"An error occurred while fetching charges by location for {dataset_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()


