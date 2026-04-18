import os
import pandas as pd
import plotly.express as px
from src.analysis.analysis import Analysis
from src.analysis.plot_utils import save_and_show_plotly


class number_of_charges_by_hour(Analysis):
    def __init__(self, id, name, info, enabled, full_custom_mode, show_images, save_images, save_results, input_interface, output_interface,
                 output_dir, data_selection, custom_params):
        super().__init__(id=id, name=name, info=info, enabled=enabled, full_custom_mode=full_custom_mode, show_images=show_images,
                         save_images=save_images, save_results=save_results, input_interface=input_interface, output_interface=output_interface,
                         output_dir=output_dir, data_selection=data_selection, custom_params=custom_params)
        return

    def check_data(self):
        # TODO check data before starting the analysis

        return

    def custom_settings(self):

        for feature, filters in self.custom_params.items():

            # Add plug in hour
            if feature == 'plug_in_hour':
                # Add feature
                if 'plug_in_hour' not in self.df.columns:
                    self.df['plug_in_datetime'] = pd.to_datetime(self.df['plug_in_datetime'])
                    self.df['plug_in_hour'] = self.df['plug_in_datetime'].dt.hour

            # Add plug out hour
            if feature == 'plug_out_hour':
                # Add feature
                if 'plug_out_hour' not in self.df.columns:
                    self.df['plug_out_datetime'] = pd.to_datetime(self.df['plug_out_datetime'])
                    self.df['plug_out_hour'] = self.df['plug_out_datetime'].dt.hour
        return

    def run(self):
        """
        # Plot total number of charging sessions by hour of the day
        """
        output_dict = {}

        # Run analysis
        for dataset_name in self.datasets_names:
            dataset_name = str(dataset_name)
            self.logger.info(f"{dataset_name} - Plot total number of charging sessions by hour of the day")

            df = self.df.loc[self.df['dataset_name'] == dataset_name]

            if df.empty:
                continue

            grouped_df_in = df.groupby(['plug_in_hour']).count()[['plug_in_datetime']]
            grouped_df_out = df.groupby(['plug_out_hour']).count()[['plug_out_datetime']]
            grouped_df = pd.concat([grouped_df_out, grouped_df_in], axis=1)
            grouped_df.rename(columns={'plug_out_datetime': 'Plug Out', 'plug_in_datetime': 'Plug In'}, inplace=True)

            fig = px.bar(
                grouped_df,
                title=f'<span style="font-size: 20px; display: block;">Number of charging sessions by hour of the day</span><br>' +
                      (f'<span style="font-size: 15px; display: block;">{self.info}</span><br>' if self.info != '' else '') +
                      f'<span style="font-size: 13px; display: block;">Dataset: {dataset_name}</span>',
                barmode='group',
                color_discrete_map={'Plug In': 'green', 'Plug Out': 'violet'}
            )
            fig.update_xaxes(title_text='Hours of the Day')
            fig.update_yaxes(title_text='Number of Charging Sessions')

            # Increase the size of text on the plot
            fig.update_layout(
                title={
                    "x": 0.5,
                    "xanchor": "center",
                    'y': 0.94,
                    'yanchor': 'top'
                },
                margin=dict(t=100),
                xaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=1
                ),
            )
            save_and_show_plotly(fig, self.save_images, self.show_images,
                                self.output_dir, f'{dataset_name}_number_of_charges_by_hour')

            output_dict[dataset_name] = grouped_df.fillna(0).astype(int).to_dict()

        self.results = output_dict

        return
