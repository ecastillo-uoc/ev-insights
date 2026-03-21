import streamlit as st
import json
import glob
import logging
from pathlib import Path

from src.__main__ import main as run_service
from src.forecast.strategies import (
    PREDICTION_TARGET_REGISTRY,
    MODEL_STRATEGY_REGISTRY
)
from src.utils.globals import CHARGING_POINT_TYPES
from src.interfaces.postgresql_interface import PostgreSql
from src.ui.shared_components import render_data_selection
import mlflow

def get_mlflow_models():
    """Fetch registered models from MLflow."""
    import requests
    import traceback
    try:
        # Try to load config
        config_path = "conf/cli/conf_mlflow_server.json"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            host = config.get('host', 'localhost')
            port = config.get('port', 5000)
            tracking_uri = f"http://{host}:{port}"
        else:
            tracking_uri = "http://localhost:5000"
            
        # Quick network check to prevent MLflowClient from hanging
        try:
            # Short timeout to see if the port is even responding
            response = requests.get(tracking_uri, timeout=2.0)
            response.raise_for_status()
        except requests.exceptions.RequestException as req_e:
            st.error(f"MLflow tracking server unreachable at {tracking_uri}: {req_e}")
            traceback.print_exc()
            return []
        
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        models = client.search_registered_models()
        return [m.name for m in models]
    except Exception as e:
        st.error(f"Exception while connecting to MLFlow: {e}")
        traceback.print_exc()
        return []

def render_forecast_page():
    st.title("📈 Forecast Service")
    
    # --- Data Selection ---
    # Use a default DB config for data selection filters to make it independent of operation mode
    default_data_config = "conf/cli/conf_cli_forecast_train_db.json"
    st.info(f"Using data source configuration: `{default_data_config}` for filter options.")
    
    selection = render_data_selection(default_data_config)

    selected_datasets = selection["datasets"]
    selected_countries = selection["countries"]
    selected_years = selection["years"]
    selected_charging_points = selection["types"]

    
    # Select Operation Mode (Train or Predict)
    mode = st.radio(
        "Operation Mode",
        ("Train", "Predict"),
        horizontal=True,
        index=0
    )
    st.markdown("---")

    # Filter config files based on naming convention
    if mode == "Train":
        config_pattern = "conf/cli/conf_cli_forecast_train*.json"
    else:
        # includes predict_query and predict_schedule
        config_pattern = "conf/cli/conf_cli_forecast_predict*.json"

    config_files = glob.glob(config_pattern)
    config_files.sort()
    
    # Select Configuration File based on Mode
    selected_config_file = st.selectbox(
        f"Select {mode} Configuration", 
        config_files,
        index=0 if config_files else None
    )
    
    # --- Configuration ---
    st.header("Forecast Configuration")
    # --- Strategy Selection (Main Page) ---
    st.subheader("Strategy Selection")

    col1, col2 = st.columns(2)

    # Prediction Target (formerly Data Strategy) Dropdown
    target_labels = {k.value: v.display_name for k, v in PREDICTION_TARGET_REGISTRY.items()}
    target_options = [None] + list(target_labels.keys())

    with col1:
        selected_target_key = st.selectbox(
            "Prediction Target",
            target_options,
            format_func=lambda x: target_labels[x] if x else "Use Config Default",
            help="Override the prediction target (data strategy) for all tasks defined in the config."
        )

    # Model Strategy Dropdown
    model_labels = {k.value: v.display_name for k, v in MODEL_STRATEGY_REGISTRY.items()}
    model_options = [None] + list(model_labels.keys())

    with col2:
        selected_model_strategy_key = st.selectbox(
            "Model Strategy",
            model_options,
            format_func=lambda x: model_labels[x] if x else "Use Config Default",
            help="Override the machine learning model strategy for all tasks defined in the config."
        )

    # If mode is Predict, allow selecting a model from MLflow
    selected_mlflow_model = None
    if mode == "Predict":
        st.subheader("Model Selection")
        with st.spinner("Fetching available models from MLflow..."):
            mlflow_models = get_mlflow_models()
        
        if mlflow_models:
            selected_mlflow_model = st.selectbox(
                "Select Trained Model",
                [None] + mlflow_models,
                help="Select a registered model from MLflow to use for prediction."
            )
        else:
            st.warning("No registered models found in MLflow (or could not connect). using config default.")

    # --- Main Area - Job Settings Display ---
    st.markdown("### ⚙️ Job Settings")
    
    # Load and display details of the selected config
    if selected_config_file:
        try:
            with open(selected_config_file, 'r') as f:
                config_json = json.load(f)
            
            # Try to extract useful info to display to the user
            # Architecture structure: service -> services -> forecast -> forecast (list)
            st.info(f"Loaded: `{selected_config_file}`")
            
            try:
                forecast_tasks = config_json.get('services', {}).get('forecast', {}).get('forecast', [])
                
                if forecast_tasks:
                    st.write("#### 📋 Tasks defined in this configuration:")
                    
                    # Create a summary table or list of tasks
                    task_summary = []
                    for task in forecast_tasks:
                        task_summary.append({
                            "ID": task.get("id"),
                            "Name": task.get("name"),
                            "Info": task.get("info"),
                            "Enabled": task.get("enabled", False)
                        })
                    
                    st.dataframe(task_summary, hide_index=True)
                    
                    # Optional: Allow user to disable specific tasks temporarily (in memory)
                    # For simplicty in V1, we just run what's in the file.
                    
                else:
                    st.warning("No forecast tasks found in this configuration file.")
                    
            except Exception as e:
                st.write("Could not parse detailed task list.")
                
        except Exception as e:
            st.error(f"Error reading config file: {e}")
    else:
        st.warning("No configuration files found matching the pattern.")

    st.markdown("---")

    # --- Execution ---
    if st.button(f"🚀 Run Forecast {mode}", type="primary", disabled=not selected_config_file):
        logging.getLogger('ui.forecast').info(
            f"[UI][Forecast] Button pressed. mode={mode}, config={selected_config_file}, "
            f"prediction_target={selected_target_key}, model_strategy={selected_model_strategy_key}, "
            f"mlflow_model={selected_mlflow_model}, "
            f"datasets={selected_datasets if selected_datasets else 'defaults'}, "
            f"countries={selected_countries if selected_countries else 'all'}, "
            f"years={selected_years if selected_years else 'all'}, "
            f"charging_points={selected_charging_points if selected_charging_points else 'all'}")
        st.info(f"Starting {mode} process...")
        
        with st.spinner("Processing... check terminal for real-time logs"):
            try:
                # 1. Prepare Configuration
                # Reload config to apply any runtime overrides
                with open(selected_config_file, 'r') as f:
                    run_config = json.load(f)
                
                # Apply strategy overrides
                overrides_applied = []
                forecast_tasks = run_config.get('services', {}).get('forecast', {}).get('forecast', [])

                if selected_charging_points:
                    overrides_applied.append(f"Charging Points: {len(selected_charging_points)} types")
                    for task in forecast_tasks:
                        if 'data_selection' not in task:
                            task['data_selection'] = {}
                        task['data_selection']['charging_point_types'] = selected_charging_points

                if selected_countries:
                    overrides_applied.append(f"Countries: {', '.join(selected_countries)}")
                    for task in forecast_tasks:
                        if 'data_selection' not in task:
                            task['data_selection'] = {}
                        task['data_selection']['countries'] = selected_countries

                if selected_years:
                    overrides_applied.append(f"Years: {len(selected_years)}")
                    for task in forecast_tasks:
                        if 'data_selection' not in task:
                            task['data_selection'] = {}
                        task['data_selection']['years'] = selected_years

                if selected_datasets:
                    overrides_applied.append(f"Datasets: {len(selected_datasets)}")
                    for task in forecast_tasks:
                        if 'data_selection' not in task:
                            task['data_selection'] = {}
                        task['data_selection']['datasets'] = selected_datasets

                if selected_target_key or selected_model_strategy_key:
                    if selected_target_key:
                        overrides_applied.append(f"Prediction Target: {target_labels[selected_target_key]}")
                        for task in forecast_tasks:
                            task['prediction_target'] = selected_target_key
                    
                    if selected_model_strategy_key:
                        overrides_applied.append(f"Model Strategy: {model_labels[selected_model_strategy_key]}")
                        for task in forecast_tasks:
                            task['model_strategy'] = selected_model_strategy_key
                    
                    # Debug: verify overrides are set on each task
                    for i, task in enumerate(forecast_tasks):
                        st.write(f"DEBUG [UI] Task {i}: name={task.get('name')}, algo={task.get('algo')}, "
                                 f"prediction_target={task.get('prediction_target')}, model_strategy={task.get('model_strategy')}")
                
                if selected_mlflow_model:
                    overrides_applied.append(f"Model Name: {selected_mlflow_model}")
                    for task in forecast_tasks:
                        task['model_name'] = selected_mlflow_model

                if overrides_applied:
                    st.write(f"ℹ️ Applying overrides: {', '.join(overrides_applied)}")

                # 2. Run Service
                # If overrides are present, pass the dictionary. Otherwise pass the file path.
                if overrides_applied:
                    result = run_service(config_json=run_config)
                else:
                    result = run_service(config_file=selected_config_file)
                
                st.success("Execution finished successfully!")
                
                # Show results if any
                if result:
                    with st.expander("Show Result Output"):
                        st.json(result)
                else:
                    st.write("No output returned (check logs).")
                    
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
                st.exception(e)
