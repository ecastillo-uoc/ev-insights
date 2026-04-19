"""Backward-compatibility shim — canonical location is ``src.data.data_fetcher``."""
from src.data.data_fetcher import *  # noqa: F401,F403
from src.data.data_fetcher import (  # explicit re-exports for type checkers
    get_engine,
    fetch_dataset_energy_trends,
    fetch_dataset_charges_trends,
    fetch_dataset_plug_ins,
    fetch_dataset_session_details_by_site,
    fetch_dataset_session_details_by_specific_site,
    fetch_daily_energy_for_forecast,
    fetch_dataset_energy_trends_by_site,
    fetch_dataset_charges_trends_by_site,
    fetch_dataset_session_details_by_station,
)
