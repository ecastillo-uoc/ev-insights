import os
import pandas as pd
import plotly.express as px
from pprint import pprint
from src.admin.admin import Admin


class delete_dataset(Admin):
    def __init__(self, id, name, info, enabled, output_interface, output_dir, save_results, custom_params):
        super().__init__(id=id, name=name, info=info, enabled=enabled, output_interface=output_interface, output_dir=output_dir,
                         save_results=save_results, custom_params=custom_params)
        return

    def run(self):
        """
        # This admin function delete all data of a specific dataset in the databse
        """
        dataset_name = self.custom_params['dataset_name']

        output_dict = {}
        results = self.output_interface.delete_dataset(dataset_name=dataset_name)
        output_dict.update(
            {
                'message': f"All data of the dataset {dataset_name} has been deleted from the database",
                'deleted': results
            }
        )

        self.logger.info(f"Dataset {dataset_name} deleted successfully from database")
        self.results = output_dict

        return self.results
