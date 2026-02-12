import streamlit as st
import json
import glob
import psycopg2
from pathlib import Path
from src.__main__ import main as run_ingestion

def render_db_manager_page():
    st.title("🛡️ DB Manager")
    
    st.sidebar.header("Admin Configuration")

    # --- DB Connectivity Check ---
    with st.expander("🔌 Database Connectivity Check"):
        st.markdown("[Open pgAdmin](http://pgadmin.GA35DX/login?next=/)")
        st.write("Test connection to PostgreSQL database")
        
        # Default values
        default_host = "localhost"
        default_port = "5432"
        default_db = "evinsights"
        default_user = "evinsights"
        default_pass = "evinsights"

        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host", value=default_host)
            port = st.text_input("Port", value=default_port)
            database = st.text_input("Database", value=default_db)
        with col2:
            user = st.text_input("User", value=default_user)
            password = st.text_input("Password", value=default_pass, type="password")
        
        if st.button("Test Connection"):
            try:
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password
                )
                conn.close()
                st.success(f"Successfully connected to {host}:{port}/{database}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # 1. Select Admin Config File
    admin_config_pattern = "conf/cli/conf_cli_admin_*.json"

    admin_config_files = glob.glob(admin_config_pattern)
    admin_config_files.sort()

    selected_admin_config = st.sidebar.selectbox(
        "Select Admin Task (Config File)",
        admin_config_files,
        index=0 if admin_config_files else None
    )

    if selected_admin_config:
        try:
            with open(selected_admin_config, 'r') as f:
                admin_config_json = json.load(f)
            
            st.subheader("Task Details")
            
            # Inspect the config to determine type and parameters
            # Assuming structure: service -> services -> admin -> admin (list)
            try:
                admin_tasks = admin_config_json.get('services', {}).get('admin', {}).get('admin', [])
                if admin_tasks:
                    task = admin_tasks[0] # Assume one task per file for simplicity
                    task_name = task.get('name')
                    task_info = task.get('info')
                    st.write(f"**Task Name:** {task_name}")
                    st.write(f"**Description:** {task_info}")
                else:
                    task_name = "Unknown"
                    st.warning("Could not find admin task definition in config.")
            except Exception as e:
                task_name = "Error"
                st.error(f"Error parsing config structure: {e}")

            # Specific UI for different tasks
            run_allowed = True
            
            if task_name == "delete_dataset":
                st.markdown("### 🗑️ Delete Dataset")
                
                # Load datasets to allow selection
                # Reusing logic to find details file - simplified for Admin
                dataset_details_file = "data/input/datasets_details.json"
                available_datasets = []
                if Path(dataset_details_file).exists():
                     with open(dataset_details_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            available_datasets = [item.get('dataset_name') for item in data if item.get('dataset_name')]
                
                dataset_to_delete = st.selectbox("Select Dataset to Delete", available_datasets)
                
                if dataset_to_delete:
                    st.warning(f"⚠️ You are about to DELETE dataset: **{dataset_to_delete}**")
                    confirm_delete = st.checkbox("I confirm I want to delete this dataset")
                    if not confirm_delete:
                        run_allowed = False
                    
                    # Update config in memory
                    if admin_tasks:
                        # Ensure custom_params exists
                        if 'custom_params' not in admin_tasks[0]:
                            admin_tasks[0]['custom_params'] = {}
                        admin_tasks[0]['custom_params']['dataset_name'] = dataset_to_delete
            
            elif task_name == "init_db":
                st.markdown("### ☢️ Initialize Database")
                st.error("⚠️ THIS WILL DELETE ALL DATA IN THE DATABASE! ⚠️")
                confirm_init = st.checkbox("I understand this will wipe the database")
                if not confirm_init:
                    run_allowed = False

            if st.button("🚀 Execute Admin Task", type="primary", disabled=not run_allowed):
                st.info(f"Executing {task_name}...")
                with st.spinner("Processing... check terminal for real-time logs"):
                    try:
                        # Run with the JSON object (modified in memory)
                        result = run_ingestion(
                            config_json=admin_config_json,
                            # datasets_list and details not needed for admin usually, 
                            # unless specific admin tasks use them via the standard interface override. 
                            # But here we updated custom_params manually.
                        )
                        st.success("Execution finished!")
                        st.subheader("Result Output:")
                        if result:
                            st.json(result)
                        else:
                            st.write("No output returned.")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.exception(e)
                        
        except Exception as e:
            st.error(f"Error loading config file: {e}")
