"""
Forecast-specific matplotlib plots.

These helpers produce the train/test-split and actual-vs-predicted
visualisations saved to ``output_plots/``.  They are called by the
forecast pipeline (``src.forecast.pipeline``) after each horizon
evaluation.
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_train_test_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_name: str,
    forecast_horizon: int,
    zoom_range=None,
    model_name: str | None = None,
    warm_up_days: int = 0,
    plots_dir: str = 'output_plots',
) -> str:
    """Plot the target variable with distinct colours for train and test periods.

    Returns the path of the saved full-range plot.

    Parameters
    ----------
    plots_dir : str
        Directory where the plot file is saved.  Defaults to ``'output_plots'``.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['y'], label='Train', linewidth=1.5, color='#1f77b4')
    plt.plot(test.index, test['y'], label='Test', linewidth=1.5, color='#ff7f0e')

    split_date = train.index.max()
    plt.axvline(x=split_date, color='grey', linestyle='--', linewidth=1, label='Train / Test split')

    if warm_up_days > 0 and warm_up_days < len(test):
        eval_start = test.index[warm_up_days]
        plt.axvspan(test.index[0], eval_start, alpha=0.12, color='orange', label='Warm-up (look_back)')
        plt.axvline(x=eval_start, color='orange', linestyle=':', linewidth=1, label='Eval start')

    title = f'Train/Test Split for {dataset_name}'
    if model_name:
        title += f'\nModel: {model_name}'
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Energy delivered (kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs(plots_dir, exist_ok=True)
    suffix = f'_{model_name}' if model_name else ''
    plot_path = os.path.join(plots_dir, f'{dataset_name}_train_test_split{suffix}_{forecast_horizon}days.png')
    plt.savefig(plot_path)
    logger.info("Train/Test split plot saved to %s", plot_path)

    if zoom_range:
        plt.xlim(pd.to_datetime(zoom_range[0]), pd.to_datetime(zoom_range[1]))
        plot_path_zoom = os.path.join(plots_dir, f'{dataset_name}_train_test_split{suffix}_{forecast_horizon}days_zoom.png')
        plt.savefig(plot_path_zoom)
        logger.info("Train/Test split zoom plot saved to %s", plot_path_zoom)

    plt.close()
    return plot_path


def plot_test_vs_predict(
    test: pd.DataFrame,
    predictions,
    dataset_name: str,
    model_name: str,
    forecast_horizon: int,
    plots_dir: str = 'output_plots',
) -> str:
    """Plot test data (actual) against predictions and save to disk.

    Parameters
    ----------
    plots_dir : str
        Directory where the plot file is saved.  Defaults to ``'output_plots'``.

    Returns
    -------
    str
        Path of the saved plot file.
    """


    min_len = min(len(test), len(predictions))
    if min_len < len(test) or min_len < len(predictions):
        logger.warning(
            "Length mismatch: test=%d, predictions=%d. Truncating to %d.",
            len(test), len(predictions), min_len,
        )
    test = test.iloc[:min_len]
    predictions = predictions[:min_len]
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['y'], label='Test (Actual)', linewidth=1.5, color='green')
    plt.plot(test.index, predictions, label='Predictions', linestyle='--', linewidth=1.5, color='red')

    # Mark weekends (Saturday=5, Sunday=6) with a small dot on the actual line
    weekends = test.index[test.index.dayofweek >= 5]
    if len(weekends):
        plt.scatter(weekends, test.loc[weekends, 'y'], color='black', s=4, zorder=5,
                    label='Weekend')

    plt.title(f'Actual vs Predictions for {dataset_name} ({forecast_horizon} days)\nModel: {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Energy Demand (kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'{dataset_name}_actual_vs_predict_{model_name}_{forecast_horizon}days.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info("Actual vs Predict plot saved to %s", plot_path)
    return plot_path


def plot_test_vs_predict_multistep(
    test: pd.DataFrame,
    windows: list,
    dataset_name: str,
    model_name: str,
    forecast_horizon: int,
    eval_strategy: str,
    plots_dir: str = 'output_plots',
    show_windows: bool = True,
    show_envelope: bool = True,
    window_sample_stride: int = 7,
) -> str:
    """Plot rolling-origin H-step predictions against test actuals.

    Each element of *windows* is a dict produced by
    ``_predict_recursive`` / ``_predict_mimo`` after inverse-transforming
    to the original kWh scale::

        {
            'origin_date':  '2021-03-05',
            'future_dates': ['2021-03-06', ..., '2021-03-12'],  # len=h
            'predictions':  [v1, ..., vh],                       # kWh
            'actuals':      [a1, ..., ah],                       # kWh
        }

    Parameters
    ----------
    test : pd.DataFrame
        Actual test series with a ``DatetimeIndex`` and ``'y'`` column
        (original kWh scale).  Used for the green actuals line.
    windows : list[dict]
        Per-origin prediction windows (see format above).
    dataset_name : str
        Dataset identifier used in the plot title and file name.
    model_name : str
        Model identifier used in the plot title and file name.
    forecast_horizon : int
        Forecast horizon in days (H).
    eval_strategy : str
        ``'recursive'`` or ``'mimo'`` — used in title and file name.
    plots_dir : str
        Directory where the plot file is saved.
    show_windows : bool
        When ``True``, overlay every *window_sample_stride*-th origin's
        H-step prediction segment as a semi-transparent blue line.
    show_envelope : bool
        When ``True``, draw a mean ± 1σ band across all windows at each
        future step offset.
    window_sample_stride : int
        Step between sampled windows when *show_windows* is ``True``.

    Returns
    -------
    str
        Path of the saved plot file.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Always: actual test series
    ax.plot(test.index, test['y'], label='Test (Actual)', linewidth=1.5, color='green', zorder=3)

    if windows:
        # Build arrays: shape (n_windows, h) aligned on future_step offset.
        # Align each window's predictions to the step offset (0..h-1).
        step_preds: dict = {}  # step_offset → list of (date, value)
        for w in windows:
            for step_i, (fd, pv) in enumerate(zip(w['future_dates'], w['predictions'])):
                step_preds.setdefault(step_i, []).append((pd.to_datetime(fd), pv))

        if show_envelope and step_preds:
            # Aggregate mean ± 1σ per calendar date over all windows
            date_to_vals: dict = {}
            for w in windows:
                for fd, pv in zip(w['future_dates'], w['predictions']):
                    date_to_vals.setdefault(pd.to_datetime(fd), []).append(pv)

            env_dates = sorted(date_to_vals.keys())
            env_mean = [float(pd.Series(date_to_vals[d]).mean()) for d in env_dates]
            env_std = [float(pd.Series(date_to_vals[d]).std(ddof=0)) for d in env_dates]

            env_mean_arr = pd.Series(env_mean, index=env_dates)
            env_std_arr = pd.Series(env_std, index=env_dates)
            ax.fill_between(
                env_dates,
                env_mean_arr - env_std_arr,
                env_mean_arr + env_std_arr,
                alpha=0.20,
                color='steelblue',
                label='Forecast mean ± 1σ',
                zorder=1,
            )
            ax.plot(env_dates, env_mean, color='steelblue', linewidth=1.0,
                    alpha=0.7, label='Forecast mean', zorder=2)

        if show_windows:
            sampled = windows[::window_sample_stride]
            for i, w in enumerate(sampled):
                dates_w = pd.to_datetime(w['future_dates'])
                preds_w = w['predictions']
                ax.plot(
                    dates_w, preds_w,
                    color='steelblue', linewidth=0.8, alpha=0.35,
                    label='Forecast windows' if i == 0 else None,
                    zorder=2,
                )

    ax.set_title(
        f'Multi-step forecast ({eval_strategy.upper()}, H={forecast_horizon}d) '
        f'for {dataset_name}\nModel: {model_name}'
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Energy Demand (kWh)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(
        plots_dir,
        f'{dataset_name}_actual_vs_predict_{model_name}_{forecast_horizon}days_{eval_strategy}.png',
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info("Multi-step predict plot saved to %s", plot_path)
    return plot_path
