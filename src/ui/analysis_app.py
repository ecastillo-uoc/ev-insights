import streamlit as st
import json
import glob
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

    # --- Prediction Visualization ---
    st.markdown("---")
    st.header("📈 Prediction Visualization")

    if not selected_config_file:
        st.warning("Select a configuration file above to enable prediction visualization.")
        return

    try:
        with open(selected_config_file, 'r') as f:
            viz_config = json.load(f)

        service_config = viz_config.get('services', {}).get('analysis', {})
        if not service_config:
            service_config = viz_config.get('services', {}).get('forecast', {})
        input_iface_cfg = service_config.get('interfaces', {}).get('input', {})

        if input_iface_cfg.get('name') != 'PostgreSql':
            st.info("Prediction visualization requires a PostgreSQL data source.")
            return

        db = PostgreSql(
            name="viz_predictions",
            type="input",
            output_dir=".",
            host=input_iface_cfg.get('host', 'localhost'),
            port=input_iface_cfg.get('port', 5432),
            user=input_iface_cfg.get('user'),
            password=input_iface_cfg.get('password'),
            database=input_iface_cfg.get('database')
        )

        col_actor, col_metric = st.columns(2)
        with col_actor:
            actor_type = st.selectbox(
                "Actor Type",
                ["charging_station", "user"],
                format_func=lambda x: "Charging Station" if x == "charging_station" else "User"
            )
        with col_metric:
            if actor_type == "charging_station":
                metric = st.selectbox("Metric", ["energy", "connections"])
            else:
                metric = st.selectbox("Metric", ["energy", "duration"])

        actor_ids = db.get_actor_ids_with_predictions(actor_type)

        if not actor_ids:
            st.info(f"No stored predictions found for {actor_type.replace('_', ' ')}s.")
            db.close_interface()
            return

        selected_actor_id = st.selectbox(
            f"Select {actor_type.replace('_', ' ').title()} ID",
            actor_ids
        )

        if st.button("📊 Load Prediction Plot"):
            with st.spinner("Fetching data..."):
                preds = db.get_predictions_history(actor_type, selected_actor_id)
                actuals = db.get_actual_daily_data(actor_type, selected_actor_id)

            if not preds:
                st.warning("No predictions found for the selected actor.")
            else:
                df_pred = pd.DataFrame(preds)
                df_pred['date'] = pd.to_datetime(df_pred['date'])
                if metric in df_pred.columns:
                    df_pred[metric] = pd.to_numeric(df_pred[metric], errors='coerce')

                df_actual = pd.DataFrame(actuals) if actuals else pd.DataFrame()
                if not df_actual.empty:
                    df_actual['date'] = pd.to_datetime(df_actual['date'])
                    if metric in df_actual.columns:
                        df_actual[metric] = pd.to_numeric(df_actual[metric], errors='coerce')

                # Build plot
                fig, ax = plt.subplots(figsize=(14, 6))

                if not df_actual.empty and metric in df_actual.columns:
                    ax.plot(df_actual['date'], df_actual[metric],
                            label="Actual", color='steelblue', linewidth=1.2)

                if metric in df_pred.columns:
                    ax.plot(df_pred['date'], df_pred[metric],
                            label="Predicted", color='orange', linestyle='--',
                            linewidth=2, marker='o', markersize=3)

                ax.set_title(
                    f"{actor_type.replace('_', ' ').title()} {selected_actor_id} — "
                    f"{metric.title()} (Actual vs Predicted)"
                )
                ax.set_xlabel("Date")
                ylabel_map = {"energy": "Energy (kWh)", "connections": "Connections",
                              "duration": "Duration (min)"}
                ax.set_ylabel(ylabel_map.get(metric, metric.title()))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                # Show data tables
                with st.expander("Prediction Data"):
                    st.dataframe(df_pred, hide_index=True)
                if not df_actual.empty:
                    with st.expander("Actual Data"):
                        st.dataframe(df_actual, hide_index=True)

        db.close_interface()

    except Exception as e:
        st.error(f"Error loading prediction visualization: {e}")
        logging.getLogger('ui.analysis').exception("Prediction visualization error")
