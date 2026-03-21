# https://ev.caltech.edu/dataset
import os
import json
from pathlib import Path
from acnportal import acndata
from datetime import datetime

def download_acn_data(api_token: str, site: str, start_time: datetime, end_time: datetime, output_path: Path):
    """
    Downloads EV charging session data from the ACN-Data API using acnportal.

    Args:
        api_token (str): The registered API token.
        site (str): The site to download data from (e.g., 'caltech', 'jpl', 'office001').
        start_time (datetime): The start time.
        end_time (datetime): The end time.
        output_path (Path): The pathlib.Path where the JSON response should be saved.
    """
    print(f"Fetching data from ACN-Data API for site '{site}'...")
    
    client = acndata.DataClient(api_token)
    
    # Use get_sessions_by_time to fetch sessions in the specified timespan
    sessions_generator = client.get_sessions_by_time(site, start=start_time, end=end_time)
    
    sessions_list = []
    
    # The generator yields each session
    for session in sessions_generator:
        sessions_list.append(session)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Store as JSON file, converting datetime objects to strings
    with open(output_path, 'w') as fh:
        json.dump({"_items": sessions_list}, fh, indent=4, default=str)
        
    print(f"Dataset for {site} successfully downloaded ({len(sessions_list)} sessions) to: {output_path}")

def main():
    api_token = os.getenv('ACN_API_KEY', 'pKEJTVnOiI9mIVU1OSYbM0aFCpW-BLVlHbSLcHHpsOs')
    
    sites = ['caltech', 'jpl', 'office001']
    
    # Start and end times as datetime objects to fetch all data available
    # ACN platform started roughly around 2018
    start_time = datetime(2018, 1, 1)
    end_time = datetime.now() # Until right now
    
    output_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'input' / 'raw_acn'
    
    for site in sites:
        output_file = output_dir / f"acndata_sessions_{site}.json"
        print(output_file)
        
        try:
            print(f"Starting download for site: {site}")
            download_acn_data(
                api_token=api_token,
                site=site,
                start_time=start_time,
                end_time=end_time,
                output_path=output_file
            )
        except Exception as e:
            print(f"Error downloading {site}: {e}")

if __name__ == '__main__':
    main()


