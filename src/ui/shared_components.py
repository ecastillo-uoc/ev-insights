import streamlit as st
import json
from pathlib import Path

from src.interfaces.postgresql_interface import PostgreSql

@st.cache_data
def fetch_db_data(config_file, datasets=None, countries=None, years=None, types=None):
    """Fetch available options and count based on hierarchical filters."""
    try:
        # If no config file is provided or it doesn't exist, return empty
        if not config_file or not Path(config_file).exists():
             return {'datasets': [], 'countries': [], 'years': [], 'types': [], 'count': 0}

        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Try analysis service first, then forecast, then generic
        service_config = config.get('services', {}).get('analysis', {})
        if not service_config:
             service_config = config.get('services', {}).get('forecast', {})
        
        input_interface = service_config.get('interfaces', {}).get('input', {})
        
        if input_interface.get('name') == 'PostgreSql':
             db_interface = PostgreSql(
                 name="temp_ui_connection",
                 type="input",
                 output_dir=".",
                 host=input_interface.get('host', 'localhost'),
                 port=input_interface.get('port', 5432),
                 user=input_interface.get('user'),
                 password=input_interface.get('password'),
                 database=input_interface.get('database')
             )
             
             # Fetch data
             res = {
                 'datasets': db_interface.get_datasets_names(),
                 'countries': db_interface.get_countries(datasets=datasets),
                 'years': db_interface.get_years(datasets=datasets, countries=countries),
                 'types': db_interface.get_charging_point_types(datasets=datasets, countries=countries, years=years),
                 'count': db_interface.get_filtered_record_count(datasets=datasets, countries=countries, years=years, types=types)
             }
             
             db_interface.close_interface()
             return res
    except Exception as e:
        # st.error(f"DB Error: {e}")
        return {'datasets': [], 'countries': [], 'years': [], 'types': [], 'count': 0}
    return {'datasets': [], 'countries': [], 'years': [], 'types': [], 'count': 0}

def render_data_selection(config_file):
    """
    Renders the data selection component with hierarchical filters.
    Returns a dictionary with selected values.
    """
    st.subheader("Data Selection")
    
    # Default empty values
    available_datasets = []
    available_countries = []
    available_years = []
    available_types = []
    record_count = 0
    
    # 1. Fetch Datasets (Level 0)
    if config_file and Path(config_file).exists():
        data_level_0 = fetch_db_data(config_file) 
        available_datasets = data_level_0.get('datasets', [])
        
    # Layout Selector
    # 1. Datasets
    selected_datasets = st.multiselect(
        "Datasets",
        available_datasets,
        help="Filter by specific datasets. Impacts Countries, Years, etc."
    )
    
    # 2. Countries (Level 1 - Depends on Datasets)
    if config_file:
        data_level_1 = fetch_db_data(config_file, datasets=selected_datasets)
        available_countries = data_level_1.get('countries', [])
        
    selected_countries = st.multiselect(
        "Countries",
        available_countries,
        default=available_countries,
        help="Filter by Country."
    )
    
    # 3. Years (Level 2 - Depends on Datasets + Countries)
    if config_file:
        data_level_2 = fetch_db_data(config_file, datasets=selected_datasets, countries=selected_countries)
        available_years = data_level_2.get('years', [])

    selected_years = st.multiselect(
        "Years",
        available_years,
        default=available_years,
        help="Filter by Year."
    )
    
    # 4. Charging Point Types (Level 3 - Depends on D+C+Y)
    if config_file:
        data_level_3 = fetch_db_data(config_file, datasets=selected_datasets, countries=selected_countries, years=selected_years)
        available_types = data_level_3.get('types', [])
        
    selected_types = st.multiselect(
        "Charging Point Types",
        available_types,
        default=available_types,
        help="Filter by Charging Point Type."
    )
    
    # 5. Count Display
    if config_file:
         data_final = fetch_db_data(config_file, 
                                    datasets=selected_datasets, 
                                    countries=selected_countries, 
                                    years=selected_years, 
                                    types=selected_types)
         record_count = data_final.get('count', 0)
    
    st.metric("Total Records Selected", f"{record_count:,}")
    
    st.markdown("---")

    return {
        "datasets": selected_datasets,
        "countries": selected_countries,
        "years": selected_years,
        "types": selected_types,
        "count": record_count
    }
