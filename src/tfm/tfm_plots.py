"""
EV charging dataset EDA plots — experiment runner.

This module is the **entry point** for generating exploratory data analysis
(EDA) visualisations used in the TFM thesis.  It iterates over configured
datasets and calls the plotting functions from ``src.analysis.dataset_plots``
to produce trend, decomposition, and box-plot figures.

All plotting logic lives in :mod:`src.analysis.dataset_plots`; this module
only handles dataset configuration and the ``main()`` loop.
"""

from pathlib import Path

from src.analysis.dataset_plots import (
    fetch_and_plot_dataset_trends,
    fetch_and_plot_session_boxplots_by_specific_site,
    fetch_and_plot_time_serie_decomposition,
)


def main():
    """Generate EDA figures for the configured datasets.

    Iterates over datasets in *li_ds*, producing energy-trend, time-series
    decomposition, and (optionally) per-site session boxplot figures.
    Output is saved to :file:`{FIG_DIR}`.
    """
    FIG_DIR = Path('/home/ecastillo/dev/M2_882_TFM/tfm/doc/figures/')

    db_config = {
        'dbname': 'evinsights',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': 5432
    }

    li_ds = ['Dundee']
    for item in li_ds:
        if isinstance(item, tuple):
            ds, site = item
        else:
            ds = item
            site = None

        print(f"\n--- Processing dataset: {ds}" + (f", site: {site}" if site else "") + " ---")

        fetch_and_plot_dataset_trends(ds, db_config, FIG_DIR, site_name=site)
        fetch_and_plot_time_serie_decomposition(ds, db_config, FIG_DIR, site_name=site)

        if site:
            fetch_and_plot_session_boxplots_by_specific_site(ds, site, db_config, FIG_DIR)


if __name__ == '__main__':
    main()
