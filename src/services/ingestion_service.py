import os
from pprint import pprint
from src.services.service import Service
from src.utils.console import Colors


class IngestionService(Service):
    def __init__(self, name, output_dir, interfaces):
        super().__init__(name=name, output_dir=output_dir, interfaces=interfaces)
        return

    def find_substring(self, list_of_strings, substring):
        return next((stringa for stringa in list_of_strings if substring in stringa), None)

    def run(self):
        self.logger.info("Ingestion service start")

        # Ingest data
        if self.input_interface.input_data_type == 'bulk':
            # Initialize counters to 0 to avoid UnboundLocalError
            dataset_count = 0
            charging_stations_count = 0
            users_count = 0
            charging_sessions_count = 0
            ev_count = 0

            # Ingest bulk datasets
            for dataset_id, dataset_value in self.input_interface.datasets.items():

                self.logger.info(f"Ingesting dataset {Colors.BLUE}{dataset_value['info']['dataset_name']}{Colors.NORMAL}" )

                # Insert dataset details
                dataset_count += self.output_interface.insert_dataset_details(data=dataset_value['info'])

                df_columns = set(dataset_value['data'].columns)
                
                self.logger.info(f"Checking columns for dataset: {dataset_value['info']['dataset_name']}")
                self.logger.info(f"DF Columns: {df_columns}")
                self.logger.info(f"Expected EV Columns: {set(self.input_interface.ev_dataframe_columns)}")

                # Determine ingestion type based on columns
                expected_ev_cols = set(self.input_interface.ev_dataframe_columns)
                expected_infra_cols = set(self.input_interface.charging_station_dataframe_columns)

                if self.input_interface.ev_dataframe_columns and expected_ev_cols.issubset(df_columns): # Use issubset to handle extra metadata cols
                     self.logger.info("Ingesting Electric Vehicles")
                     ev_count += self.output_interface.insert_electric_vehicles(data=dataset_value['data'])
                     
                elif self.input_interface.charging_station_dataframe_columns and expected_infra_cols.issubset(df_columns):
                     self.logger.info("Ingesting Charging Stations (Infrastructure)")
                     # TODO: Implement insert_charging_stations_only if needed or reuse existing
                     # For now, re-using existing but it might expect session columns? Let's check.
                     # Existing expects 'orig_id', 'manufacturer', etc. Our current infrastructure df has 'origin_id', 'ocpp_version'...
                     # This needs alignment. I will direct to a new method.
                     charging_stations_count += self.output_interface.insert_infrastructure(data=dataset_value['data'], dataset_name=dataset_value['info']['dataset_name'])

                else:
                    # Default: Sessions ingestion (which implicitly inserts stations and users)
                    self.logger.info("Ingesting Charging Sessions")
                    
                    # Insert charging stations details (extracted from session data usually)
                    charging_stations_count += self.output_interface.insert_charging_stations(
                        data=dataset_value['data'],
                        dataset_name=dataset_value['info']['dataset_name']
                    )

                    # Insert users
                    users_count += self.output_interface.insert_users(
                        data=dataset_value['data'],
                        dataset_name=dataset_value['info']['dataset_name']
                    )

                    # Insert charging sessions
                    charging_sessions_count += self.output_interface.insert_charging_sessions(
                        data=dataset_value['data'],
                        dataset_name=dataset_value['info']['dataset_name']
                    )

                # Add here further ingestion if needed
                # ...

            self.output.append({'Message': "Bulk data ingested successfully"})
            self.output.append(
                {'Counts': {'dataset_count': dataset_count,
                            'charging_stations_count': charging_stations_count,
                            'users_count': users_count,
                            'charging_sessions_count': charging_sessions_count,
                            'ev_count': ev_count}
                 }
            )

        elif self.input_interface.input_data_type == 'table':
            # Ingest single tables
            # Ingest tables in this order
            tables = ["Dataset", "User", "ChargingStation", "ChargingSession"]
            for table in tables:
                dataset_path = self.find_substring(self.input_interface.data.keys(), table)
                if dataset_path:
                    dataset_value = self.input_interface.data[dataset_path]

                    self.logger.info(f"Ingesting file: {dataset_value['file_name']}")

                    # TODO it works, but check if this is the proper way to handle the injection of the same entry in dataset table (probably this is ok)
                    if dataset_value['table'] == "Dataset":
                        self.logger.info(f"Ingesting Dataset")
                        try:
                            self.output_interface.insert_dataset_details(data=dataset_value['df'].to_dict('records')[0])
                            self.output.append("Dataset ingested successfully")
                        except Exception as e:
                            self.output.append(str(e))
                            break

                    elif dataset_value['table'] == "User":
                        self.logger.info(f"Ingesting Users")
                        self.output_interface.insert_users(data=dataset_value['df'],
                                                           dataset_name=dataset_value['pilot'])
                        self.output.append("Users ingested successfully")

                    elif dataset_value['table'] == "ChargingStation":
                        self.logger.info(f"Ingesting ChargingStation")
                        self.output_interface.insert_charging_stations(data=dataset_value['df'],
                                                                       dataset_name=dataset_value['pilot'])
                        self.output.append("ChargingStations ingested successfully")

                    elif dataset_value['table'] == "ChargingSession":
                        self.logger.info(f"Ingesting ChargingSession")
                        self.output_interface.insert_charging_sessions(data=dataset_value['df'],
                                                                       dataset_name=dataset_value['pilot'])
                        self.output.append("ChargingSessions ingested successfully")

        self.logger.info("Ingestion service end")


        return self.output
