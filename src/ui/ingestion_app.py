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

st.title("⚡ EV Insights Ingestion")

# --- Configuration Section ---
st.sidebar.header("Ingestion configuration")

# 1. Select Ingestion Type (Bulk or Table)
ingestion_type = st.sidebar.radio(
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

selected_config_file = st.sidebar.selectbox(
    f"Select configuration file ({ingestion_type})", 
    config_files,
    index=0 if config_files else None
)

# 3. Select Datasets configuration details file
# Scan for Datasets configuration details config files
dataset_config_files = glob.glob("data/input/*.json")
# Sort to make it reproducible
config_files.sort()

dataset_configs = [f for f in dataset_config_files]
default_details_file = "data/input/datasets_details.json"
# details_file = st.sidebar.text_input("Datasets configuration details file", value=default_details_file)
selected_details_file = st.sidebar.selectbox(
    "Select Datasets configuration details file", 
    dataset_configs,
    index=0 if dataset_configs else None
)

# --- Datasets Selection ---
st.header("Select Datasets")

# Load available datasets
available_datasets = []
if selected_details_file is None:
    # Handle the error case appropriately for your app
    raise ValueError("selected_details_file cannot be None")

path_obj = Path(selected_details_file) 
if path_obj.exists():
    try:
        with open(selected_details_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Assuming list of dicts with 'dataset_name'
            if isinstance(data, list):
                available_datasets = [item.get('dataset_name') for item in data if item.get('dataset_name')]
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
else:
    st.markdown("**Selected Datasets:** *Reading from Config File default*")

if st.button("🚀 Run Ingestion", type="primary"):
    if not selected_config_file:
        st.error("Please select a configuration file.")
    else:
        st.info("Starting process...")
        
        with st.spinner("Processing... check terminal for real-time logs"):
            try:
                # Prepare arguments
                datasets_arg = selected_datasets if selected_datasets else None
                
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
