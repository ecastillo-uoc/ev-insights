"""Backward-compatibility shim — canonical location is ``src.analysis.dataset_plots``."""
from src.analysis.dataset_plots import *  # noqa: F401,F403
from src.analysis.dataset_plots import (  # explicit re-exports for type checkers
    fetch_and_plot_charges_by_month,
    fetch_and_plot_charges_by_weekday,
    fetch_and_plot_charges_time_serie_decomposition,
    fetch_and_plot_charges_trends_by_site,
    fetch_and_plot_covid_patterns,
    fetch_and_plot_dataset_charges_trends,
    fetch_and_plot_dataset_trends,
    fetch_and_plot_energy_by_month,
    fetch_and_plot_energy_by_weekday,
    fetch_and_plot_energy_trends_by_site,
    fetch_and_plot_multiple_datasets_trends,
    fetch_and_plot_session_boxplots_by_site,
    fetch_and_plot_session_boxplots_by_specific_site,
    fetch_and_plot_session_boxplots_by_station,
    fetch_and_plot_time_serie_decomposition,
    plot_charges_by_month,
    plot_charges_by_weekday,
    plot_charges_by_weekday_covid,
    plot_charges_trends,
    plot_charges_trends_by_site,
    plot_energy_by_month,
    plot_energy_by_weekday,
    plot_energy_by_weekday_covid,
    plot_energy_consumption_trends,
    plot_energy_consumption_trends_by_site,
    plot_multiple_energy_consumption_trends,
    plot_session_boxplots_by_site,
    tag_covid_period,
)
from pathlib import Path


def main():
    """Main execution function (kept for backward compatibility)."""
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
