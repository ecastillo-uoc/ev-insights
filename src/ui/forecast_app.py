import streamlit as st
import json
import glob
from pathlib import Path
from src.__main__ import main as run_service
from src.forecast.strategies import (
    PREDICTION_TARGET_REGISTRY,
    MODEL_STRATEGY_REGISTRY
)

def render_forecast_page():
    st.title("📈 Forecast Service")
    
    # --- Sidebar Configuration ---
    st.sidebar.header("Forecast Configuration")
    
    # 1. Select Operation Mode (Train or Predict)
    mode = st.sidebar.radio(
        "Operation Mode",
        ("Train", "Predict"),
        index=0
    )
    
    # 2. Select Configuration File based on Mode
    # Filter config files based on naming convention
    if mode == "Train":
        config_pattern = "conf/cli/conf_cli_forecast_train*.json"
    else:
        # includes predict_query and predict_schedule
        config_pattern = "conf/cli/conf_cli_forecast_predict*.json"

    config_files = glob.glob(config_pattern)
    config_files.sort()
    
    selected_config_file = st.sidebar.selectbox(
        f"Select {mode} Configuration", 
        config_files,
        index=0 if config_files else None
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Strategy Overrides")
    
    # Prediction Target (formerly Data Strategy) Dropdown
    target_labels = {k.value: v.display_name for k, v in PREDICTION_TARGET_REGISTRY.items()}
    target_options = [None] + list(target_labels.keys())
    
    selected_target_key = st.sidebar.selectbox(
        "Prediction Target",
        target_options,
        format_func=lambda x: target_labels[x] if x else "Use Config Default",
        help="Override the prediction target (data strategy) for all tasks defined in the config."
    )
    
    # Model Strategy Dropdown
    model_labels = {k.value: v.display_name for k, v in MODEL_STRATEGY_REGISTRY.items()}
    model_options = [None] + list(model_labels.keys())
    
    selected_model_strategy_key = st.sidebar.selectbox(
        "Model Strategy",
        model_options,
        format_func=lambda x: model_labels[x] if x else "Use Config Default",
        help="Override the machine learning model strategy for all tasks defined in the config."
    )
    
    # --- Main Area ---
    
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
        st.info(f"Starting {mode} process...")
        
        with st.spinner("Processing... check terminal for real-time logs"):
            try:
                # 1. Prepare Configuration
                # Reload config to apply any runtime overrides
                with open(selected_config_file, 'r') as f:
                    run_config = json.load(f)
                
                # Apply strategy overrides
                overrides_applied = []
                if selected_target_key or selected_model_strategy_key:
                    forecast_tasks = run_config.get('services', {}).get('forecast', {}).get('forecast', [])
                    if selected_target_key:
                        overrides_applied.append(f"Prediction Target: {target_labels[selected_target_key]}")
                        # Update all tasks
                        for task in forecast_tasks:
                             task['prediction_target'] = selected_target_key
                    
                    if selected_model_strategy_key:
                        overrides_applied.append(f"Model Strategy: {model_labels[selected_model_strategy_key]}")
                        # Update all tasks
                        for task in forecast_tasks:
                             task['model_strategy'] = selected_model_strategy_key
                    
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
