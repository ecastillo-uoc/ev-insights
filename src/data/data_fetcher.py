"""
Data access layer (DAL) for EV charging session data.

Provides functions that fetch data from the PostgreSQL ``evinsights``
database and return pandas DataFrames ready for analysis or forecasting.

The module is **framework-agnostic** — it has no dependency on any ML or
thesis-specific code.  It is consumed by:

* ``src.forecast.pipeline`` — daily energy for forecast models
* ``src.analysis.dataset_plots`` — EDA / thesis-chapter visualisations
"""

import logging

import pandas as pd
from sqlalchemy import create_engine

from src.data.constants import COVID_START, COVID_END, EXCLUDE_COVID_DATA

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

    engine_url = (
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )
    return create_engine(engine_url)


def fetch_dataset_energy_trends(dataset_name: str, db_config: dict = None, site_name: str = None) -> pd.DataFrame:
    """
    Fetches the dataset by name from the database and returns the energy trends as a DataFrame.
    Admits optional site_name filter.
    """
    query = """
    SELECT 
        DATE(cs.plug_out_datetime) as timestamp, 
        SUM(cs.energy_supplied) as energy_kwh,
        COUNT(DISTINCT cs.fk_charging_station_id) as station_count,
        EXTRACT(EPOCH FROM AVG(
            CASE
                WHEN charge_end_datetime - plug_in_datetime >= INTERVAL '0' THEN charge_end_datetime - plug_in_datetime
                WHEN plug_out_datetime - plug_in_datetime >= INTERVAL '0' THEN plug_out_datetime - plug_in_datetime
                ELSE INTERVAL '0'
            END
        )) as avg_session_duration_s,
        COUNT(DISTINCT cs.id) as session_count
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
        logger.error("Error fetching %s: %s", dataset_name, e)
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
        logger.error("Error fetching charges for %s: %s", dataset_name, e)
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
        logger.error("Error fetching plug_in data for %s: %s", dataset_name, e)
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
        logger.error("Error fetching session details for %s: %s", dataset_name, e)
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
        logger.error("Error fetching specific site details for %s at %s: %s", dataset_name, site_name, e)
        return pd.DataFrame()


def fetch_daily_energy_for_forecast(dataset_name: str, db_config: dict = None,
                                    exclude_covid: bool = None) -> pd.DataFrame:
    """
    Fetches daily energy demand data and formats it for forecasting models.
    Fills missing days with 0 and removes the COVID no-data period.
    Returns DataFrame with a DatetimeIndex 'ds' and values 'y'.

    Parameters
    ----------
    exclude_covid : bool | None
        If ``True``/``False``, override the global ``EXCLUDE_COVID_DATA``
        constant.  ``None`` (default) defers to the constant.
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
    should_exclude = exclude_covid if exclude_covid is not None else EXCLUDE_COVID_DATA
    if should_exclude:
        covid_mask = (df.index >= pd.to_datetime(COVID_START)) & (df.index <= pd.to_datetime(COVID_END))
        df = df[~covid_mask]

    return df


def _warn_zero_gaps(df: pd.DataFrame, col: str, dataset_name: str,
                    min_consecutive: int = 14) -> None:
    """Log a warning when the series contains long runs of consecutive zeros."""
    is_zero = (df[col] == 0).astype(int)
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
        logger.error("Error fetching by site for %s: %s", dataset_name, e)
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
        logger.error("Error fetching charges by site for %s: %s", dataset_name, e)
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
        engine = get_engine(db_config)
        from sqlalchemy import text
        with engine.connect() as conn:
            dataset = pd.read_sql(text(query), conn, params=params)
        return dataset
    except Exception as e:
        logger.error("Error fetching specific station details: %s", e)
        return pd.DataFrame()
