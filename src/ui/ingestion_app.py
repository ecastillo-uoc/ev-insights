import streamlit as st
import sys
import json
import glob
from pathlib import Path

# Add project root to path so we can import src
# assuming this script is run from ev-insights/ but lives in src/ui/
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.__main__ import main as run_ingestion
except ImportError:
    st.error("Could not import 'src'. Make sure you are running this app from the 'ev-insights' directory.")
    st.stop()

st.set_page_config(page_title="EV Insights Ingestion", layout="wide")

st.title("⚡ EV Insights Ingestion Runner")

# --- Configuration Section ---
st.sidebar.header("Ingestion configuration")

# 1. Select Configuration File
# Scan for ingestion compliant config files
config_files = glob.glob("conf/cli/conf_cli_ingestion*.json")
# Sort to make it reproducible
config_files.sort()

# Filter for ingestion related configs if desired, or let user pick any
ingestion_configs = [f for f in config_files if "ingestion" in f]
other_configs = [f for f in config_files if "ingestion" not in f]

selected_config_file = st.sidebar.selectbox(
    "Select configuration file", 
    ingestion_configs + other_configs, 
    index=0 if ingestion_configs else None
)

# 2. Select Datasets configuration details file
default_details_file = "data/input/datasets_details.json"
details_file = st.sidebar.text_input("Datasets configuration details file", value=default_details_file)

# --- Datasets Selection ---
st.header("Select Datasets")

# Load available datasets
available_datasets = []
if Path(details_file).exists():
    try:
        with open(details_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Assuming list of dicts with 'dataset_name'
            if isinstance(data, list):
                available_datasets = [item.get('dataset_name') for item in data if item.get('dataset_name')]
    except Exception as e:
        st.error(f"Error reading details file: {e}")
else:
    st.warning(f"Details file not found at: {details_file}")

# Config specific hints
datasets_help = "Choose datasets to ingest. If you leave this empty, the defaults defined inside the JSON config file will be used."
selected_datasets = st.multiselect(
    "Overrides 'datasets_list'",
    options=available_datasets,
    default=None,
    help=datasets_help
)

# --- Execution ---
st.markdown("---")
if st.button("🚀 Run Ingestion", type="primary"):
    if not selected_config_file:
        st.error("Please select a configuration file.")
    else:
        st.info(f"Starting process with config: `{selected_config_file}`...")
        
        status_container = st.container()
        
        # Capture stdout/stderr/logs is tricky within the same process without threading issues,
        # but since main() runs synchronously, we can try to rely on the return value 
        # or just display a spinner.
        
        with st.spinner("Processing... check terminal for real-time logs"):
            try:
                # Prepare arguments
                datasets_arg = selected_datasets if selected_datasets else None
                
                # Call the main function directly
                result = run_ingestion(
                    config_file=selected_config_file,
                    datasets_list=datasets_arg,
                    datasets_details_file=details_file
                )
                
                st.success("Execution finished successfully!")
                
                st.subheader("Result Output:")
                if result:
                    for item in result:
                        st.write(item)
                else:
                    st.write("No output returned (check logs).")

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
                st.exception(e)
