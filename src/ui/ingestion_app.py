import streamlit as st
import json
import glob
import logging
from pathlib import Path
from src.__main__ import main as run_ingestion

logger = logging.getLogger('ui.ingestion')

def render_ingestion_page():
    st.title("⚡ EV Insights Ingestion")

    # --- Configuration Section ---
    st.header("Ingestion configuration")

    # 1. Select Ingestion Type (Bulk or Table)
    ingestion_type = st.radio(
        "Ingestion Type",
        ("bulk", "table"),
        index=0
    )

    # 2. Select Configuration File based on Type
    # Scan for ingestion compliant config files
    if ingestion_type == "bulk":
        config_pattern = "conf/cli/conf_cli_ingestion_bulk*.json"
    else:
        config_pattern = "conf/cli/conf_cli_ingestion_table*.json"

    config_files = glob.glob(config_pattern)
    # Sort to make it reproducible
    config_files.sort()

    selected_config_file = st.selectbox(
        f"Select configuration file ({ingestion_type})", 
        config_files,
        index=0 if config_files else None
    )

    # 3. Select Datasets configuration details file
    # Scan for Datasets configuration details config files
    dataset_config_files = glob.glob("data/input/*.json")
    # Sort to make it reproducible
    dataset_config_files.sort()

    dataset_configs = [f for f in dataset_config_files]
    preferred_details_file = "data/input/datasets_details_ok.json"
    default_details_index = (
        dataset_configs.index(preferred_details_file)
        if preferred_details_file in dataset_configs
        else (0 if dataset_configs else None)
    )
    selected_details_file = st.selectbox(
        "Select Datasets configuration details file", 
        dataset_configs,
        index=default_details_index
    )

    # --- Datasets Selection ---
    st.header("Select Datasets")

    # Load available datasets
    available_datasets = []
    if selected_details_file is None:
        # Handle the error case appropriately for your app
        # raise ValueError("selected_details_file cannot be None")
        pass
    else:
        path_obj = Path(selected_details_file) 
        if path_obj.exists():
            try:
                with open(selected_details_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Assuming list of dicts with 'dataset_name'
                    if isinstance(data, list):
                        available_datasets = sorted(
                            [item.get('dataset_name') for item in data if item.get('dataset_name')],
                            key=str.casefold,
                        )
            except Exception as e:
                st.error(f"Error reading details file: {e}")
        else:
            st.warning(f"Details file not found: {selected_details_file}")

    # Config specific hints
    datasets_help = "Choose datasets to ingest. If you leave this empty, the defaults defined inside the JSON config file will be used."
    selected_datasets = st.multiselect(
        "Overrides 'datasets_list'",
        options=available_datasets,
        default=None,
        help=datasets_help
    )

    # --- Trigger Execution ---
    st.markdown("---")
    # Show summary of parameters
    st.subheader("Execution Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Ingestion Type:** `{ingestion_type}`")
    with col2:
        st.markdown(f"**Config File:** `{selected_config_file}`")
    with col3:
        st.markdown(f"**Details File:** `{selected_details_file}`")

    if selected_datasets:
        st.markdown(f"**Selected Datasets ({len(selected_datasets)}):** {', '.join(selected_datasets)}")

        # Show file count preview for datasets with glob patterns
        if selected_details_file and selected_config_file:
            try:
                with open(selected_config_file, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                service_name = cfg.get('service', '')
                input_cfg = cfg.get('services', {}).get(service_name, {}).get('interfaces', {}).get('input', {})
                input_dir = input_cfg.get('input_dir', '')
                if input_dir:
                    input_dir_path = Path(input_dir)

                with open(selected_details_file, 'r', encoding='utf-8') as f:
                    details_data = json.load(f)

                for ds_name in selected_datasets:
                    ds_info = next((d for d in details_data if d.get('dataset_name') == ds_name), None)
                    if ds_info:
                        file_pattern = ds_info.get('dataset_file_name', '')
                        folder = ds_info.get('dataset_directory', ds_info.get('dataset_folder', ds_name))
                        folder_path = input_dir_path / folder
                        exact_file = folder_path / file_pattern
                        if exact_file.is_file():
                            st.info(f"📁 **{ds_name}**: 1 file (`{file_pattern}`)")
                        else:
                            matched = sorted(folder_path.glob(file_pattern)) if folder_path.is_dir() else []
                            if matched:
                                file_names = [f.name for f in matched]
                                st.info(f"📁 **{ds_name}**: {len(matched)} file(s) matching `{file_pattern}`")
                                with st.expander(f"Files for {ds_name}"):
                                    for fname in file_names:
                                        st.text(f"  • {fname}")
                            elif not folder_path.is_dir():
                                st.warning(f"📁 **{ds_name}**: folder not found (`{folder_path}`)")
                            else:
                                st.warning(f"📁 **{ds_name}**: no files matching `{file_pattern}` in `{folder_path}`")
            except Exception as e:
                logger.debug(f"Could not resolve file counts: {e}")
    else:
        st.markdown("**Selected Datasets:** *None selected - Using defaults from Config File*")

    if st.button("🚀 Run Ingestion", type="primary"):
        if not selected_config_file:
            st.error("Please select a configuration file.")
        else:
            logger.info(f"[UI][Ingestion] Button pressed. type={ingestion_type}, "
                        f"config={selected_config_file}, details={selected_details_file}, "
                        f"datasets={selected_datasets or 'defaults'}")
            st.info("Starting process...")
            
            with st.spinner("Processing... check terminal for real-time logs"):
                try:
                    # Prepare arguments
                    # Explicitly pass the selected list. If empty (None or []), main() might us defaults if passed None.
                    # We pass 'selected_datasets' which is a list. If it's empty [], main overrides config with [].
                    # If we want defaults, we pass None.
                    
                    datasets_arg = selected_datasets if selected_datasets else None
                    if datasets_arg:
                        st.write(f"Overriding config with datasets: {datasets_arg}")
                    else:
                        st.write("Using default datasets from configuration file.")

                    # Call the main function directly
                    result = run_ingestion(
                        config_file=selected_config_file,
                        datasets_list=datasets_arg,
                        datasets_details_file=selected_details_file
                    )
                    
                    st.success("Execution finished successfully!")
                    
                    st.subheader("Result Output:")
                    if result:
                        st.json(result)
                    else:
                        st.write("No output returned (check logs).")

                except Exception as e:
                    st.error(f"An error occurred during execution: {e}")
                    st.exception(e)


