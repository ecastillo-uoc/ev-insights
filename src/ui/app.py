import streamlit as st
import sys
from pathlib import Path

# Add project root to path so we can import src
# assuming this script is run from ev-insights/ but lives in src/ui/
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import pages
try:
    from src.ui.ingestion_app import render_ingestion_page
    from src.ui.db_manager_app import render_db_manager_page
    from src.ui.forecast_app import render_forecast_page
except ImportError:
    # If standard import fails (e.g. if we are running from src/ui directly), try adjusting path
    # But sys.path.append above should handle it if running from root.
    # If running "streamlit run src/ui/app.py" from root, "src.ui.ingestion_app" should work.
    try:
        from ingestion_app import render_ingestion_page
        from db_manager_app import render_db_manager_page
        from forecast_app import render_forecast_page
    except ImportError as e:
         st.error(f"Could not import modules: {e}")
         st.stop()


st.set_page_config(page_title="EV Insights Manager", layout="wide")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Ingestion", "DB Manager", "Forecast"])

if page == "Ingestion":
    render_ingestion_page()
elif page == "DB Manager":
    render_db_manager_page()
elif page == "Forecast":
    render_forecast_page()
