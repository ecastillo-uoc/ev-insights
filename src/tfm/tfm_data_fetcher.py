import logging
import pandas as pd
from sqlalchemy import create_engine

from src.tfm.tfm_constants import COVID_START, COVID_END, EXCLUDE_COVID_DATA

logger = logging.getLogger(__name__)

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

def fetch_dataset_energy_trends(dataset_name: str, db_config: dict = None, site_name: str = None) -> pd.DataFrame:
    """
    Fetches the dataset by name from the database and returns the energy trends as a DataFrame.
    Admits optional site_name filter.
    """
    query = """
    SELECT 
        DATE(cs.plug_out_datetime) as timestamp, 
        SUM(cs.energy_supplied) as energy_kwh
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d 
        ON cs.fk_dataset_id = d.id
    """
    params = {'dataset_name': dataset_name}
    if site_name:
        query += """
    JOIN evinsights."ChargingStation" st
        ON cs.fk_charging_station_id = st.id
    WHERE d.name = %(dataset_name)s AND st.site_name = %(site_name)s
    """
        params['site_name'] = site_name
    else:
        query += """
    WHERE d.name = %(dataset_name)s
    """

    query += """
    GROUP BY DATE(cs.plug_out_datetime)
    ORDER BY timestamp
    """
    try:       
        engine = get_engine(db_config)
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params=params)
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

def fetch_dataset_session_details_by_site(dataset_name: str, db_config: dict = None) -> pd.DataFrame:
    """
    Fetches the dataset by name from the database and returns session details including
    plug_in_datetime, plug_out_datetime, energy_supplied, and site.
    """
    query = """
    SELECT 
        cs.plug_in_datetime, 
        cs.plug_out_datetime,
        cs.energy_supplied,
        st.site_name AS site
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d 
        ON cs.fk_dataset_id = d.id
    JOIN evinsights."ChargingStation" st
        ON cs.fk_charging_station_id = st.id
    WHERE d.name = %(dataset_name)s
    """
    try:
        engine = get_engine(db_config)
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        return dataset
    except Exception as e:
        print(f"An error occurred while fetching session details for {dataset_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()

def fetch_dataset_session_details_by_specific_site(dataset_name: str, site_name: str, db_config: dict = None) -> pd.DataFrame:
    """
    Fetches the dataset by name and specific site from the database and returns session details.
    """
    query = """
    SELECT 
        cs.plug_in_datetime, 
        cs.plug_out_datetime,
        cs.energy_supplied,
        cs.plug_in_soc,
        st.site_name AS site,
        st.station_name,
        d.name AS dataset_name
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d 
        ON cs.fk_dataset_id = d.id
    JOIN evinsights."ChargingStation" st
        ON cs.fk_charging_station_id = st.id
    WHERE d.name = %(dataset_name)s AND st.site_name = %(site_name)s
    """
    try:
        engine = get_engine(db_config)
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name, 'site_name': site_name})
        return dataset
    except Exception as e:
        print(f"An error occurred while fetching specific site details for {dataset_name} at {site_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()

def fetch_daily_energy_for_forecast(dataset_name: str, db_config: dict = None) -> pd.DataFrame:
    """
    Fetches daily energy demand data and formats it for forecasting models (Prophet/LSTM).
    Fills missing days with 0 and removes the COVID no-data period.
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
    
    # EDA: detect long consecutive-zero stretches that signal data gaps
    _warn_zero_gaps(df, 'y', dataset_name)

    # Remove COVID no-data period — the gap contains only artificial zeros
    # from asfreq fill, not real observations
    if EXCLUDE_COVID_DATA:
        covid_mask = (df.index >= pd.to_datetime(COVID_START)) & (df.index <= pd.to_datetime(COVID_END))
        df = df[~covid_mask]
    
    return df


def _warn_zero_gaps(df: pd.DataFrame, col: str, dataset_name: str,
                    min_consecutive: int = 14) -> None:
    """Log a warning when the series contains long runs of consecutive zeros.

    This helps identify data gaps (e.g. COVID no-data period) that were filled
    with artificial zeros by ``asfreq``/``reindex``.
    """
    is_zero = (df[col] == 0).astype(int)
    # Group consecutive identical values; keep only the zero groups
    groups = is_zero.ne(is_zero.shift()).cumsum()
    for _, grp in df[is_zero == 1].groupby(groups):
        if len(grp) >= min_consecutive:
            logger.warning(
                "Dataset '%s': %d consecutive zero-value days detected "
                "from %s to %s. This may indicate a data gap "
                "(e.g. COVID no-data period). Consider setting "
                "EXCLUDE_COVID_DATA = True.",
                dataset_name, len(grp),
                grp.index.min().strftime('%Y-%m-%d'),
                grp.index.max().strftime('%Y-%m-%d'),
            )

def fetch_dataset_energy_trends_by_site(dataset_name: str, db_config: dict = None) -> pd.DataFrame:
    """
    Fetches the dataset by name from the database and returns the energy trends, grouped by site (site_name).
    """
    query = """
    SELECT 
        DATE(cs.plug_out_datetime) as timestamp, 
        st.site_name AS site,
        SUM(cs.energy_supplied) as energy_kwh
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d 
        ON cs.fk_dataset_id = d.id
    JOIN evinsights."ChargingStation" st
        ON cs.fk_charging_station_id = st.id
    WHERE d.name = %(dataset_name)s
    GROUP BY DATE(cs.plug_out_datetime), st.site_name
    ORDER BY timestamp
    """
    try:       
        engine = get_engine(db_config)
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        return dataset
    except Exception as e:
        print(f"An error occurred while fetching by site for {dataset_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()

def fetch_dataset_charges_trends_by_site(dataset_name: str, db_config: dict = None) -> pd.DataFrame:
    """
    Fetches the dataset by name from the database and returns the daily session 
    counts grouped by site (site_name).
    """
    query = """
    SELECT 
        DATE(cs.plug_in_datetime) as timestamp, 
        st.site_name AS site,
        COUNT(cs.id) as sessions_count
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."Dataset" d 
        ON cs.fk_dataset_id = d.id
    JOIN evinsights."ChargingStation" st
        ON cs.fk_charging_station_id = st.id
    WHERE d.name = %(dataset_name)s
    GROUP BY DATE(cs.plug_in_datetime), st.site_name
    ORDER BY timestamp
    """
    try:       
        engine = get_engine(db_config)
        with engine.connect() as conn:
            dataset = pd.read_sql(query, conn, params={'dataset_name': dataset_name})
        return dataset
    except Exception as e:
        print(f"An error occurred while fetching charges by site for {dataset_name}: Query: {query}\n -- {e}")
        return pd.DataFrame()



def fetch_dataset_session_details_by_station(station_ids: tuple = None, city: str = None, db_config: dict = None) -> pd.DataFrame:
    """
    Fetches session details for specific charging stations, optionally filtered by city or station_ids or both.
    """
    query = """
    SELECT 
        cs.plug_in_datetime, 
        cs.plug_out_datetime,
        cs.energy_supplied,
        cs.plug_in_soc,
        st.station_name AS site,
        st.city
    FROM 
        evinsights."ChargingSession" cs
    JOIN evinsights."ChargingStation" st
        ON cs.fk_charging_station_id = st.id
    WHERE 1=1
    """
    params = {}
    if station_ids:
        query += " AND cs.fk_charging_station_id IN %(station_ids)s"
        params['station_ids'] = tuple(station_ids)
    if city:
        query += " AND st.city = %(city)s"
        params['city'] = city
        
    try:
        #print(query)
        engine = get_engine(db_config)
        from sqlalchemy import text
        with engine.connect() as conn:
            dataset = pd.read_sql(text(query), conn, params=params)
        return dataset
    except Exception as e:
        print(f"An error occurred while fetching specific station details: {e}")
        return pd.DataFrame()
