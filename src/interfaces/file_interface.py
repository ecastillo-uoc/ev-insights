import os
import json
import glob
import logging
import csv
from pathlib import Path
import pandas as pd
from pprint import pprint
from datetime import datetime, timedelta
from dateutil import parser

from src.interfaces.interface import Interface
from src.utils.db_config import get_dataframe_columns_from_db
from src.utils.constants import CONNECTOR_TYPE_ALIASES, DEFAULT_EV_DATAFRAME_COLUMNS, DEFAULT_CHARGING_STATION_DATAFRAME_COLUMNS
from src.utils.console import Colors
from src.utils.path_utils import normalize_path

_logger = logging.getLogger("interfaces.file")


class File(Interface):
    """
    A class to handle file-based data ingestion and processing for datasets.

    Attributes:
        input_dir (str): Directory containing input files.
        limit_rows (int): Maximum number of rows to read from each dataset.
        input_data_type (str): Type of input data ('bulk' or 'table').
        datasets_details_file (str): Path to the file containing dataset metadata.
        datasets_list (list): List of dataset names to process.
        datasets (dict): Dictionary containing processed datasets.
        df (pd.DataFrame): Combined DataFrame of all processed datasets.
    """
    def __init__(self, name, type, input_dir, output_dir, limit_rows, input_data_type, datasets_list, datasets_details_file):
        """
        Initializes the File interface for data ingestion and processing.

        Args:
            name (str): Name of the interface.
            type (str): Type of the interface ('input' or 'output').
            input_dir (str): Directory containing input files.
            output_dir (str): Directory for output files.
            limit_rows (int): Maximum number of rows to read from each dataset.
            input_data_type (str): Type of input data ('bulk' or 'table').
            datasets_list (list): List of dataset names to process.
            datasets_details_file (str): Path to the file containing dataset metadata.
        """
        super().__init__(name=name, type=type, output_dir=output_dir)
        self.logger = _logger
        
        if input_dir is not None:
            self.input_dir = normalize_path(input_dir)
        else:
            self.input_dir = None

        self.limit_rows = limit_rows
        self.input_data_type = input_data_type
        self.dataframe_columns = get_dataframe_columns_from_db()
        self.ev_dataframe_columns = DEFAULT_EV_DATAFRAME_COLUMNS
        self.charging_station_dataframe_columns = DEFAULT_CHARGING_STATION_DATAFRAME_COLUMNS

        # Data gathering from files
        if self.type == "input":
            if isinstance(datasets_details_file, list):
                 self.datasets_details_file = [normalize_path(f) for f in datasets_details_file]
            else:
                self.datasets_details_file = normalize_path(datasets_details_file)
            self.datasets_list = datasets_list

            if self.input_data_type == 'bulk':
                # Get bulk dataset
                self.datasets = self.get_datasets_from_files(datasets_list=datasets_list,
                                                             datasets_details_file=self.datasets_details_file,
                                                             limit_rows=self.limit_rows)
                self.datasets = self.prepare_datasets()
                self.df = self.convert_datasets_in_df()

            elif self.input_data_type == 'table':
                # Get data structured in tables (dataset, user, chargingstation, chargingsession)
                self.logger.info(f"Ingest files from this folder {self.input_dir}")
                self.file_list = list(Path(self.input_dir).glob("*.csv"))

                self.data = {}
                for file_path in self.file_list:
                    file_name = file_path.name
                    self.data.update({file_path: {'file_name': file_name,
                                                  'pilot': file_name.split("_")[1] if len(file_name.split("_")) == 3 else file_name.split("_")[0],
                                                  'table': (file_name.split("_")[2] if len(file_name.split("_")) == 3 else file_name.split("_")[1]).split(".")[0],
                                                  'df': pd.read_csv(filepath_or_buffer=file_path, delimiter=",", encoding='utf-8')}})
            else:
                raise Exception(f"Wrong input_data_type ({self.input_data_type}). Please specify 'bulk' or 'table'")

        return

    def close_interface(self):
        """
        Placeholder method for closing the interface.
        """
        return

    def init_db(self):
        pass  # File interface does not manage a database

    def delete_dataset(self, dataset_name):
        raise NotImplementedError("File interface does not support delete_dataset")

    def get_model(self, algo, model_name, models_dir=None):
        """Delegate model loading to model_persistence."""
        from src.forecast.model_persistence import load_model
        if models_dir is None:
            return None, None, None
        model = load_model(algo=algo, model_name=model_name, models_dir=models_dir)
        return None, None, model

    def save_forecast_prediction(self, forecaster_name, actor, model, experiment_id, run_id, results, algo):
        raise NotImplementedError("File interface does not support save_forecast_prediction")

    def sort_list(self, file_path):
        """
        Sorts a list of files based on a priority map.

        Args:
            file_path (str): Path to the file.

        Returns:
            int: Priority value for sorting.
        """
        file_name = Path(file_path).name.lower()
        for key in self.priority_map:
            if key in file_name:
                return self.priority_map[key]
        return float('inf')

    def convert_datasets_in_df(self):
        """
        Converts all datasets into a single DataFrame.

        Returns:
            pd.DataFrame: Combined DataFrame of all datasets.
        """
        df = pd.DataFrame()
        for dataset_id, dataset_value in self.datasets.items():

            # Add data to df
            df_tmp = dataset_value['data']

            # Add info to df
            i = 0
            for key in dataset_value['info'].keys():
                df_tmp.insert(i, key, dataset_value['info'][key])
                i += 1

            df = pd.concat([df, df_tmp]).reset_index(drop=True)

        return df

    def get_datasets_from_files(self, datasets_list, datasets_details_file, limit_rows=None):
        """
        Reads datasets from files based on metadata and returns them as a dictionary.

        Args:
            datasets_list (list): List of dataset names to process.
            datasets_details_file (str or list): Path to the file(s) containing dataset metadata.
            limit_rows (int, optional): Maximum number of rows to read.

        Returns:
            dict: Dictionary containing dataset metadata and data.
        """
        datasets_details = []
        files_to_read = datasets_details_file if isinstance(datasets_details_file, list) else [datasets_details_file]

        for file_path in files_to_read:
            # Check if file_path is a pure windows/posix path object or a string and handle accordingly
            current_file_path = file_path if isinstance(file_path, Path) else Path(file_path)

            if not current_file_path.exists():
                self.logger.warning(f"Metadata file not found: {current_file_path}")
                continue

            if current_file_path.suffix == '.json':
                with open(current_file_path, mode='r', encoding='utf-8') as file:
                    data = json.load(file)
                    datasets_details.extend([row for row in data if row['dataset_name'] in datasets_list])
            else:
                with open(current_file_path, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    datasets_details.extend([row for row in reader if row['dataset_name'] in datasets_list])
        
        datasets = {}
        for dataset_info in datasets_details:
            self.logger.info(f"Gathering {Colors.BLUE}{dataset_info['dataset_name']}{Colors.NORMAL} dataset")
            # Use 'dataset_directory' if specified, otherwise 'dataset_folder', otherwise 'dataset_name'
            folder_name = dataset_info.get('dataset_directory', dataset_info.get('dataset_folder', dataset_info['dataset_name']))
            dataset_name = dataset_info.get('dataset_file_name')
            if folder_name is None or dataset_name is None:
                raise ValueError("Currently processing path requires both dataset_directory/folder and dataset_name")

            # Type checker now knows they are definitely strings
            input_file = Path(self.input_dir) / folder_name / dataset_name

            self.logger.info(input_file)

            if input_file.is_file():
                df_orig = pd.DataFrame()

                dataset_type = dataset_info.get('dataset_file_type')
                self.logger.info(f"Dataset type: {Colors.YELLOW}{dataset_type}{Colors.NORMAL}") 

                if dataset_type == 'csv':
                    _delimiter = dataset_info.get('dataset_delimiter')
                    _encoding = dataset_info.get('dataset_encoding')
                    _text_quotes = dataset_info.get('text_quotes','"').replace('\'','')
                    _decimal = dataset_info.get('decimal',',')
                    
                    read_csv_args = {
                        "filepath_or_buffer": input_file,
                        "delimiter": _delimiter,
                        "encoding": _encoding,
                        "nrows": limit_rows,
                        "decimal": _decimal
                    }
                    if _text_quotes:
                        read_csv_args["quotechar"] = _text_quotes
                    else:
                        read_csv_args["quoting"] = csv.QUOTE_NONE

                    df_orig = pd.read_csv(**read_csv_args)

                if dataset_type == 'xlsx':
                    df_orig = pd.read_excel(io=input_file, sheet_name=dataset_info['dataset_sheet_name'],
                                            engine='openpyxl', nrows=limit_rows)

                if dataset_type == 'json':
                    try:
                        data = json.loads(input_file.read_text())
                        # TODO: Generalize "items" json flattening may be an issue
                        df_orig = pd.json_normalize(data, '_items').head(self.limit_rows)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse JSON file {input_file}: {e}")
                        continue # Skip this dataset and continue
                
                if df_orig is not None:
                    datasets.update({dataset_info['dataset_name']: {"info": dataset_info,
                                                                  "data": df_orig}})
            else:
                raise Exception("File '%s' does not exist. Please check dataset name '%s' and file name '%s'" %
                                (input_file, dataset_info['dataset_name'], dataset_info.get('dataset_file_name')))

        return datasets

    def get_data(self, data_selection: dict):
        """
        Retrieves data based on the specified selection criteria.

        Args:
            data_selection (dict): Dictionary specifying datasets and fields to retrieve.

        Returns:
            pd.DataFrame: Filtered DataFrame based on the selection criteria.
        """
        # Get list of available datasets
        datasets_available_names = self.df['dataset_name'].unique()

        # Fill datasets_list in case datasets param is missing or void
        datasets_names_filter = data_selection['datasets'] \
            if ('datasets' in data_selection and len(data_selection['datasets']) > 0) else datasets_available_names

        for datasets_name in datasets_names_filter:
            if datasets_name not in datasets_available_names:
                raise Exception("Dataset '%s' is not available. Available datasets: %s" % (datasets_name, datasets_available_names))

        # Get list of available fields
        fields_available_names = self.df.columns

        # Fill fields in case field param is missing or void
        datasets_fields_filter = data_selection['fields'] \
            if ('fields' in data_selection and len(data_selection['fields']) > 0) else fields_available_names

        # Check columns existence
        for field in datasets_fields_filter:
            if field not in fields_available_names:
                raise Exception("Field '%s' is not available. Available fields: %s" % (field, fields_available_names))

        # Add dataset name columns in any case
        datasets_fields_filter['dataset_name'] = None if 'dataset_name' not in datasets_fields_filter else None

        # Check merge_dataset param and unify data in a single dataset "Dataset_merged" if true
        if data_selection['merge_datasets']:
            self.df['dataset_name'] = "Dataset_merged"

        return self.df.loc[:, datasets_fields_filter.keys()]

    def convert_date(self, date_string):
        """
        Converts a date string with a shortened year format to a full datetime object.

        Args:
            date_string (str): The input date string with a shortened year (e.g., '23-01-01').

        Returns:
            pd.Timestamp: The converted datetime object.
        """
        # Rimuovi i primi due caratteri dall'anno e aggiungi "20" prima di convertire
        date_string_with_2000 = "20" + date_string[2:]
        # Converti la stringa di data in formato datetime
        return pd.to_datetime(date_string_with_2000, format='%Y-%m-%d %H:%M:%S')

    def prepare_dataset_amb_barcelona_ev(self, df):
        _logger.info(f"Ingestion: Preparing {Colors.BLUE}AMB_Barcelona_EV{Colors.NORMAL} dataset")
        _logger.debug(df.columns.to_list())

        # normalize names
        df.rename(columns={
            "Marchio":"ev_manufacturer",
            "Modello":"ev_model",
            "BatteriaE (kWh)":"ev_battery_capacity_kWh",
        }, inplace=True)
        
        # Filter columns
        df = df[self.ev_dataframe_columns]

        return df
    

    def prepare_dataset_ev_infra(self, df):
        _logger.info(f"Ingestion: Preparing {Colors.BLUE}AMB_Barcelona_EV / EV_models{Colors.NORMAL} dataset")
        _logger.info(f"Columns found: {df.columns.to_list()}")

        # normalize names
        df.rename(columns={
            "make":"ev_manufacturer",
            "model":"ev_model",
        }, inplace=True)
        
        # Filter columns
        df["ev_battery_capacity_kWh"] = ""
        df = df[self.ev_dataframe_columns]

        return df

    def prepare_dataset_amb_barcelona_charging_stations(self, df):
        _logger.info(f"Ingestion: Preparing {Colors.BLUE}AMB_Barcelona_charging_points{Colors.NORMAL} dataset")
        _logger.debug(df.columns.to_list())

        # normalize names
        df.rename(columns={
            "CHARGING POINT (name and adress)": "origin_id",
            "OCPP version": "ocpp_version",
            "longitude"	:"longitude",
            "latitude": "latitude",
            "Plug type (AC 22 kW/DC 50 kW)" : "connector",
            "Cumulative energy delivered in the year (Wh)": "energy_year_Wh",
            "Average charge power (W)": "power_W_avg"
        }, inplace=True)

        # Handle missing columns that might not be in the CSV
        if "energy_year_Wh" not in df.columns:
            df["energy_year_Wh"] = None
        if "power_W_avg" not in df.columns:
            df["power_W_avg"] = None

        # Logic to infer connector if not present
        if "connector" not in df.columns:
            possible_connector_columns = [
                "Schuko 3kW 16A mode 1", 
                "Mennekes 7 kW 16A mode 3", 
                "Mennekes 43 kW 63A mode 3", 
                "CHAdeMO 55kW 125A mode 4", 
                "COMBO CCS 55 kW 125A mode 4"
            ]
            
            # Find which of these columns actually exist in df (handling duplicates if pandas mangled them like .1)
            existing_conn_cols = []
            for col in df.columns:
                # Check directly or starts with (due to pandas duplicate handling)
                for p_col in possible_connector_columns:
                    if col == p_col or col.startswith(p_col + "."):
                        existing_conn_cols.append((col, p_col))
                        break
            
            def get_connector_string(row):
                connectors = []
                for col_name, conn_type in existing_conn_cols:
                    if str(row[col_name]).lower() == 'x':
                        connectors.append(conn_type)
                return ", ".join(sorted(list(set(connectors)))) if connectors else None

            df["connector"] = df.apply(get_connector_string, axis=1)
        
        # Filter columns
        df = df[self.charging_station_dataframe_columns]

        return df

    def prepare_dataset_amb_barcelona(self, df):
        """
        Prepares the dataset for the 'AMB_Barcelona' source by renaming columns and adding required fields.

        Args:
            df (pd.DataFrame): The input DataFrame containing raw data.

        Returns:
            pd.DataFrame: The processed DataFrame with standardized columns and additional fields.
        """
        _logger.info(f"Ingestion: Preparing {Colors.BLUE}BeLib{Colors.NORMAL} dataset")
        _logger.debug(df.columns.to_list())

        # normalize names
        df.rename(columns={
            'CHARGING POINT': 'orig_ds',
            'CONNECTOR':'connector',
            'START TIME':'plug_in_datetime',
            'STOP TIME':'plug_out_datetime',
            'DURATION (min)': 'duration_min',
            'CONSUMPTION (kWh)': 'energy_supplied',
            'CHARGING POINT': 'charging_station_id',
            'Pmax': 'max_charging_power'
            }, inplace=True)
        
        # map connector aliases
        if 'connector' in df.columns:
            df['connector'] = df['connector'].map(CONNECTOR_TYPE_ALIASES).fillna(df['connector'])

        # process data
        df['plug_in_datetime'] = (pd.to_datetime(df['plug_in_datetime'], dayfirst=True).astype('string') + " " +
                                  df['ora'].astype('string') + ":" + df['Mins'].astype('string'))
        df['plug_in_datetime'] = pd.to_datetime(df['plug_in_datetime'], format='%Y-%m-%d %H:%M')
        df['charge_end_datetime'] = df['plug_in_datetime'] + df['duration_min'].apply(lambda x: timedelta(minutes=x))
        df['plug_out_datetime'] = df['charge_end_datetime']
        df['charge_end_datetime_presence'] = False
        df['user_id'] = pd.NA
        df['ev_id'] = pd.NA
        df['ev_max_charging_power'] = pd.NA
        df = df[self.dataframe_columns]
        
        return df

    def prepare_dataset_olev(self, df):
        """
        Prepares the dataset for the 'OLEV' source by renaming columns and adding required fields.

        Args:
            df (pd.DataFrame): The input DataFrame containing raw data.

        Returns:
            pd.DataFrame: The processed DataFrame with standardized columns and additional fields.
        """
        _logger.info(f"Ingestion: Preparing {Colors.BLUE}BeLib{Colors.NORMAL} dataset")
        _logger.debug(df.columns.to_list())
        df.rename(columns={'Energy': 'energy_supplied',
                           'CPID': 'charging_station_id'}, inplace=True)
        df['plug_in_datetime'] = pd.to_datetime(df.StartDate + " " + df.StartTime, format='%Y-%m-%d %H:%M:%S')
        df['charge_end_datetime'] = pd.to_datetime(df.EndDate + " " + df.EndTime, format='%Y-%m-%d %H:%M:%S')
        df['plug_out_datetime'] = df['charge_end_datetime']
        df['charge_end_datetime_presence'] = False
        df['user_id'] = pd.NA
        df['max_charging_power'] = pd.NA
        df['ev_id'] = pd.NA
        df['ev_max_charging_power'] = pd.NA
        df = df[self.dataframe_columns]
        return df

    def prepare_dataset_harvard_dataverse(self, df):
        """
        Prepares the dataset for the 'Harvard Dataverse' source by renaming columns and adding required fields.

        Args:
            df (pd.DataFrame): The input DataFrame containing raw data.

        Returns:
            pd.DataFrame: The processed DataFrame with standardized columns and additional fields.
        """
        _logger.info(f"Ingestion: Preparing {Colors.BLUE}BeLib{Colors.NORMAL} dataset")
        _logger.debug(df.columns.to_list())
        df.rename(columns={'kwhTotal': 'energy_supplied',
                           'stationId': 'charging_station_id',
                           'userId': 'user_id'}, inplace=True)
        df['plug_in_datetime'] = df['created'].apply(lambda x: self.convert_date(x))
        df['charge_end_datetime'] = df['ended'].apply(lambda x: self.convert_date(x))
        df['plug_out_datetime'] = df['charge_end_datetime']
        df['charge_end_datetime_presence'] = False
        df['max_charging_power'] = pd.NA
        df['ev_id'] = pd.NA
        df['ev_max_charging_power'] = pd.NA
        df = df[self.dataframe_columns]
        return df

    def prepare_dataset_Elaad(self, df):
        """
        Prepares the dataset for the 'Elaad' source by renaming columns and adding required fields.

        Args:
            df (pd.DataFrame): The input DataFrame containing raw data.

        Returns:
            pd.DataFrame: The processed DataFrame with standardized columns and additional fields.
        """
        df.rename(columns={'TotalEnergy': 'energy_supplied',
                           'MaxPower': 'ev_max_charging_power',
                           'StartCard': 'user_id'},
                  inplace=True)
        df['plug_in_datetime'] = pd.to_datetime(df.UTCTransactionStart, dayfirst=True)
        df['charge_end_datetime'] = pd.to_datetime(df.UTCTransactionStop, dayfirst=True)
        df['plug_out_datetime'] = df['charge_end_datetime']
        df['charge_end_datetime_presence'] = False
        df['charging_station_id'] = df.apply(lambda row: '{}_{}'.format(row['ChargePoint'], row['Connector']), axis=1)
        df['ev_id'] = pd.NA
        df['max_charging_power'] = pd.NA
        df = df[self.dataframe_columns]
        return df

    def prepare_dataset_belib(self, df):
        """
        Prepares the dataset for the 'BeLib' source by renaming columns and adding required fields.

        Args:
            df (pd.DataFrame): The input DataFrame containing raw data.

        Returns:
            pd.DataFrame: The processed DataFrame with standardized columns and additional fields.

            ID PDC local;Statut du point de recharge;URL Description Point de charge;
            Heure mise à jour;coordonneesXY;adresse_station;code_insee_commune;
            arrondissement
        """
        _logger.info(f"Ingestion: Preparing {Colors.BLUE}BeLib{Colors.NORMAL} dataset")
        _logger.debug(df.columns.to_list())
        df.rename(columns={'Prise de courant': 'max_charging_power',
                           'Borne': 'charging_station_id',
                           'UUID Badge': 'user_id'},
                  inplace=True)
        df['plug_in_datetime'] = df['Date de début'].apply(lambda x: pd.to_datetime(parser.parse(x))).dt.tz_localize(None)
        df['charge_end_datetime'] = df['Date de fin'].apply(lambda x: pd.to_datetime(parser.parse(x))).dt.tz_localize(None)
        df['plug_out_datetime'] = df['charge_end_datetime']
        df['charge_end_datetime_presence'] = False
        df['energy_supplied'] = df["L'énergie (Wh)"] / 1000
        df['ev_id'] = pd.NA
        df['ev_max_charging_power'] = pd.NA
        df = df[self.dataframe_columns]
        return df

    def prepare_dataset_ACN(self, df) -> pd.DataFrame:
        """
        Prepares the dataset for the 'ACN_Caltech' source by renaming columns and adding required fields.

        Args:
            df (pd.DataFrame): The input DataFrame containing raw data.

        Returns:
            pd.DataFrame: The processed DataFrame with standardized columns and additional fields.
        """
        df.rename(columns={
                        # _id
                        # clusterID
                        # connectionTime
                        # disconnectTime
                        # doneChargingTime
                        "kWhDelivered": "energy_supplied",
                        # sessionID
                        # siteID
                        # spaceID
                        "stationID": "charging_station_id",
                        # timezone
                        "userID": "user_id"
                        # userInputs
                           },
                  inplace=True)
        df['plug_in_datetime'] = pd.to_datetime(df.connectionTime)
        df['charge_end_datetime'] = pd.to_datetime(df.doneChargingTime)
        df['plug_out_datetime'] = pd.to_datetime(df.disconnectTime)
        df['charge_end_datetime_presence'] = True
        df['max_charging_power'] = pd.NA
        df['ev_id'] = pd.NA
        df['ev_max_charging_power'] = pd.NA
        df = df[self.dataframe_columns]

        return df

    def prepare_dataset_Norway_12loc(self, df):
        """
        Prepares the dataset for the 'Norway_12loc' source by renaming columns and adding required fields.

        Args:
            df (pd.DataFrame): The input DataFrame containing raw data.

        Returns:
            pd.DataFrame: The processed DataFrame with standardized columns and additional fields.
        """
        _logger.info(f"Ingestion: Preparing {Colors.BLUE}BeLib{Colors.NORMAL} dataset")
        _logger.debug(df.columns.to_list())
        df.rename(columns={"energy_session": "energy_supplied"},
                  inplace=True)
        df['charging_station_id'] = df.user_id  # in this dataset the charging_station_id is missing, since the charging stations are private we can use the user_id
        df['plug_in_datetime'] = pd.to_datetime(df.plugin_time)
        df['charge_end_datetime'] = pd.to_datetime(df.plugout_time)
        df['plug_out_datetime'] = df['charge_end_datetime']
        df['charge_end_datetime_presence'] = False
        df['max_charging_power'] = pd.NA
        df['ev_id'] = pd.NA
        df['ev_max_charging_power'] = pd.NA
        df = df[self.dataframe_columns]
        return df

    # Here you can add other datasets
    # def prepare_dataset_<new_dataset_name>(self, df):
    #     ...
    #     return df

    def check_dataset_columns(self, df):
        """
        Validates that the DataFrame contains the expected columns for one of the supported types.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if the DataFrame contains the expected columns for one supported type.

        Raises:
            ValueError: If the DataFrame does not contain the expected columns for any supported type.
        """
        columns = set(df.columns)
        
        # Check against Session columns
        if columns == set(self.dataframe_columns):
            return True
            
        # Check against EV columns
        if columns == set(self.ev_dataframe_columns):
            return True
            
        # Check against Charging Station columns
        if columns == set(self.charging_station_dataframe_columns):
            return True
        
        # If none match
        raise ValueError("Columns are not properly set for any known type\n"
                         "Expected Session columns: %s\n"
                         "Expected EV columns: %s\n"
                         "Expected Station columns: %s\n"
                         "df.columns=%s"
                         % (self.dataframe_columns, self.ev_dataframe_columns, 
                            self.charging_station_dataframe_columns, df.columns.tolist()))


    def prepare_datasets(self):
        """
        Prepares datasets by applying specific transformations based on dataset names.

        Returns:
            dict: Dictionary of transformed datasets.
        """
        new_datasets = {}
        for dataset_id, dataset_value in self.datasets.items():
            dataset_name = dataset_value['info']['dataset_name']
            prepare_method = self._preparation_methods.get(dataset_name)

            if prepare_method:
                df = prepare_method(df=dataset_value['data'])
            else:
                raise ValueError("Wrong dataset name: %s" % dataset_name)

            # Check dataset columns
            if self.check_dataset_columns(df):
                if 'plug_in_datetime' in df.columns:
                    df = df.sort_values(by='plug_in_datetime')
                new_datasets.update({dataset_id: {"info": dataset_value['info'], "data": df}})

        return new_datasets

    def save_forecast_model(self, forecaster_name, results, file_path, algo=None):
        """Delegate to model_persistence (kept for backward compatibility)."""
        from src.forecast.model_persistence import save_model
        save_model(forecaster_name=forecaster_name, results=results,
                   models_dir=str(Path(file_path).parent), algo=algo)


    @property
    def _preparation_methods(self):
        """
        Returns a mapping of dataset names to their corresponding preparation methods.
        To add a new dataset:
        1. Implement the prepare_dataset_<name> method.
        2. Add the mapping here.
        """
        return {
            # EV
            'EV_models': self.prepare_dataset_ev_infra,
            'AMB_Barcelona_EV': self.prepare_dataset_amb_barcelona_ev,

            # Charging points
            'AMB_Barcelona_charging_points': self.prepare_dataset_amb_barcelona_charging_stations,

            # Charging sessions
            'AMB_Barcelona': self.prepare_dataset_amb_barcelona,
            'BeLib': self.prepare_dataset_belib,
            'ACN_Caltech': self.prepare_dataset_ACN,
            'ACN_JPL': self.prepare_dataset_ACN,
            'ACN_Office001': self.prepare_dataset_ACN,
            'Norway_12loc': self.prepare_dataset_Norway_12loc,
            'Elaad': self.prepare_dataset_Elaad,
            'Harvard_dataverse': self.prepare_dataset_harvard_dataverse,
            'OLEV': self.prepare_dataset_olev,

            # 'Hanse_und_Universitaetsstadt_Rostock': self.prepare_dataset_hanse_und_universitaetsstadt_rostock, 
        }