import streamlit as st
import json
import glob
import logging
from pathlib import Path

from src.__main__ import main as run_service
# from src.utils.globals import CHARGING_POINT_TYPES  # Not used anymore
from src.interfaces.postgresql_interface import PostgreSql
from src.ui.shared_components import render_data_selection

def render_analysis_page():
    st.title("📊 Analysis Service")
    
    # --- Configuration ---
    st.header("Analysis Configuration")
    
    # Select Configuration File
    config_pattern = "conf/cli/conf_cli_analysis*.json"
    config_files = glob.glob(config_pattern)
    config_files.sort()
    
    selected_config_file = st.selectbox(
        "Select Configuration", 
        config_files,
        index=0 if config_files else None
    )

    # --- Data Selection ---
    selection = render_data_selection(selected_config_file)
    selected_datasets = selection["datasets"]
    selected_countries = selection["countries"]
    selected_years = selection["years"]
    selected_types = selection["types"]
    
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
        logging.getLogger('ui.analysis').info(
            f"[UI][Analysis] Button pressed. config={selected_config_file}, "
            f"selected_tasks={selected_analysis_names}, "
            f"datasets={selected_datasets if selected_datasets else 'defaults'}, "
            f"countries={selected_countries if selected_countries else 'all'}, "
            f"years={selected_years if selected_years else 'all'}, "
            f"types={selected_types if selected_types else 'all'}")
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
