import streamlit as st
import json
import glob
from pathlib import Path

from src.__main__ import main as run_service
# from src.utils.globals import CHARGING_POINT_TYPES  # Not used anymore
from src.interfaces.postgresql_interface import PostgreSql

@st.cache_data
def fetch_db_data(config_file, datasets=None, countries=None, years=None, types=None):
    """Fetch available options and count based on hierarchical filters."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        service_config = config.get('services', {}).get('analysis', {})
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

def render_analysis_page():
    st.title("📊 Analysis Service")
    
    # --- Sidebar Configuration ---
    st.sidebar.header("Analysis Configuration")
    
    # Select Configuration File
    config_pattern = "conf/cli/conf_cli_analysis*.json"
    config_files = glob.glob(config_pattern)
    config_files.sort()
    
    selected_config_file = st.sidebar.selectbox(
        "Select Configuration", 
        config_files,
        index=0 if config_files else None
    )

    # --- Data Selection ---
    st.subheader("Data Selection")
    
    # Default empty values
    available_datasets = []
    available_countries = []
    available_years = []
    available_types = []
    record_count = 0
    
    # 1. Fetch Datasets (Level 0)
    if selected_config_file and Path(selected_config_file).exists():
        data_level_0 = fetch_db_data(selected_config_file) 
        available_datasets = data_level_0['datasets']
        
    # Layout Selector - Rearranged
    # 1. Datasets
    selected_datasets = st.multiselect(
        "1. Datasets",
        available_datasets,
        help="Filter by specific datasets. Impacts Countries, Years, etc."
    )
    
    # 2. Countries (Level 1 - Depends on Datasets)
    if selected_config_file:
        data_level_1 = fetch_db_data(selected_config_file, datasets=selected_datasets)
        available_countries = data_level_1['countries']
        
    selected_countries = st.multiselect(
        "2. Countries",
        available_countries,
        default=available_countries,
        help="Filter by Country."
    )
    
    # 3. Years (Level 2 - Depends on Datasets + Countries)
    if selected_config_file:
        data_level_2 = fetch_db_data(selected_config_file, datasets=selected_datasets, countries=selected_countries)
        available_years = data_level_2['years']

    selected_years = st.multiselect(
        "3. Years",
        available_years,
        default=available_years,
        help="Filter by Year."
    )
    
    # 4. Charging Point Types (Level 3 - Depends on D+C+Y)
    if selected_config_file:
        data_level_3 = fetch_db_data(selected_config_file, datasets=selected_datasets, countries=selected_countries, years=selected_years)
        available_types = data_level_3['types']
        
    selected_types = st.multiselect(
        "4. Charging Point Types",
        available_types,
        default=available_types,
        help="Filter by Charging Point Type."
    )
    
    # 5. Count Display
    if selected_config_file:
         data_final = fetch_db_data(selected_config_file, 
                                    datasets=selected_datasets, 
                                    countries=selected_countries, 
                                    years=selected_years, 
                                    types=selected_types)
         record_count = data_final['count']
    
    st.metric("Total Records Selected", f"{record_count:,}")

    st.markdown("---")
    
    # --- Main Area ---
    st.subheader("⚙️ Analysis Catalogue & Execution")
    
    selected_analysis_names = []
    
    if selected_config_file:
        try:
            with open(selected_config_file, 'r') as f:
                config_json = json.load(f)
            
            # st.info(f"Loaded: `{selected_config_file}`")
            
            try:
                analysis_tasks = config_json.get('services', {}).get('analysis', {}).get('analysis', [])
                
                if analysis_tasks:
                    # Map names to tasks for easy lookup
                    task_map = {task.get("name"): task for task in analysis_tasks}
                    task_names = list(task_map.keys())
                    
                    selected_analysis_names = st.multiselect(
                        "5. Select Analysis Tasks to Run",
                        task_names,
                        help="Choose one or more analysis tasks from the catalogue to execute on the selected data."
                    )
                    
                    if selected_analysis_names:
                        st.write("Selected Tasks:")
                        for name in selected_analysis_names:
                            task = task_map[name]
                            st.caption(f"- **{name}**: {task.get('info', 'No description available')}")
                    
                else:
                    st.warning("No analysis tasks found in this configuration file.")
                    
            except Exception as e:
                st.error("Could not parse detailed task list.")
                
        except Exception as e:
            st.error(f"Error reading config file: {e}")
    else:
        st.warning("No configuration files found.")

    st.markdown("---")
    
    # --- Execution ---
    if st.button("🚀 Run Analysis", type="primary", disabled=not (selected_config_file and selected_analysis_names)):
        st.info("Starting analysis process...")
        
        result_placeholder = st.empty()
        
        with st.spinner("Processing... check terminal for real-time logs"):
            try:
                # 1. Prepare Configuration with Overrides
                with open(selected_config_file, 'r') as f:
                    run_config = json.load(f)
                
                overrides_applied = []
                # Filter to only keep selected tasks and enable them
                original_tasks = run_config.get('services', {}).get('analysis', {}).get('analysis', [])
                filtered_tasks = []
                
                for task in original_tasks:
                    if task.get("name") in selected_analysis_names:
                        task['enabled'] = True # Ensure it is enabled
                        filtered_tasks.append(task)
                
                # Update config with filtered list
                run_config['services']['analysis']['analysis'] = filtered_tasks
                
                analysis_tasks = filtered_tasks # Reference for data overrides loop below
                
                # Apply Data Selection Overrides
                if selected_types:
                    overrides_applied.append(f"Charging Points: {len(selected_types)} types")
                    for task in analysis_tasks:
                        if 'data_selection' not in task:
                            task['data_selection'] = {}
                        task['data_selection']['charging_point_types'] = selected_types

                if selected_countries:
                    overrides_applied.append(f"Countries: {', '.join(selected_countries)}")
                    for task in analysis_tasks:
                         if 'data_selection' not in task:
                            task['data_selection'] = {}
                         task['data_selection']['countries'] = selected_countries

                if selected_years:
                     overrides_applied.append(f"Years: {len(selected_years)}")
                     for task in analysis_tasks:
                         if 'data_selection' not in task:
                            task['data_selection'] = {}
                         task['data_selection']['years'] = selected_years

                if selected_datasets:
                     overrides_applied.append(f"Datasets: {len(selected_datasets)}")
                     for task in analysis_tasks:
                         if 'data_selection' not in task:
                            task['data_selection'] = {}
                         task['data_selection']['datasets'] = selected_datasets

                if overrides_applied:
                    st.write(f"ℹ️ Applying overrides: {', '.join(overrides_applied)}")

                # 2. Run Service
                result = run_service(config_json=run_config)
                
                st.success("Analysis execution finished successfully!")
                
                with result_placeholder.container():
                     st.write("### Analysis Results")
                     if result:
                        st.json(result)
                     else:
                        st.info("Check the `data/output` folder or logs for results if they are not returned here.")
                    
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
                st.exception(e)
