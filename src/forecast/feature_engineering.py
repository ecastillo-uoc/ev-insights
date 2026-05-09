"""
Feature engineering and target transforms for the forecast pipeline.

This module centralises all feature-construction logic that is shared across
model strategies, keeping the pipeline orchestrator (``src.forecast.pipeline``) and the
individual strategy files free of feature-level implementation details.

Three orthogonal capabilities — each controlled by a boolean flag in
``strategy_params`` — can be combined freely, yielding up to 8 experiment
configurations:

+---------------------+------------------------------------------------------+
| Flag                | Effect                                               |
+=====================+======================================================+
| use_log_transform   | ``np.log1p`` / ``np.expm1`` round-trip on target     |
+---------------------+------------------------------------------------------+
| use_differencing    | 7-day seasonal differencing / per-date reconstruction|
+---------------------+------------------------------------------------------+
| use_calendar_features | Day-of-week sin/cos encoding and business day      |
|                     | indicator — added to the DataFrame                   |
+---------------------+------------------------------------------------------+

**Ordering rules**

* Forward transforms:  log **first** → diff **second**
  (diff of log-values gives log-returns — a well-known stationary transform).
* Inverse transforms:  undo diff **first** → undo log **second**
  (strict reverse of the forward order; swapping corrupts the result).

**7-day seasonal differencing**

``use_differencing`` applies a lag-7 difference (``y[d] - y[d-7]``), removing
weekly seasonality instead of just yesterday's level.  The first 7 rows are
dropped to avoid NaN propagation.  Reconstruction uses the actual value from
7 days prior (stored in ``df_before_diff``).

**Why per-date reconstruction instead of cumsum?**

Cumulative sum propagates prediction error: if prediction 1 is off by +100
every subsequent prediction inherits that bias.  Per-date lookup uses the
*actual* previous-7-day value (from ``df_before_diff``), keeping each
prediction's error independent.  The cumsum fallback is only used in
schedule mode where actual previous values are unavailable.

**Per-window normalisation (RevIN-style)**

``use_window_norm`` standardises the target using the rolling mean and std
computed over the preceding ``window_norm_days`` days.  Each window's
normalisation stats are stored in ``state.revin_stats`` and used during
inverse-transform to recover the original scale.  This removes the local
level and scale from each prediction window without requiring a global
scaler, reducing distribution-shift sensitivity.

Normalisation is only applied once a full ``window_norm_days`` window of
prior data is available (``min_periods=window_norm_days``).  Rows with
insufficient history receive the identity transform (mu=0, sigma=1) so
they pass through unchanged and can be correctly inverted.  A tiny
absolute floor (``1e-6``) prevents division by zero for truly constant
windows without distorting real data.

**Calendar features — dtype choice**

Active calendar columns are ``day_of_week_sin``, ``day_of_week_cos``, and
``is_business_day`` (the annual sin/cos day-of-year columns are retained in
the code but commented out — day-of-week and the business-day flag capture
the dominant weekly pattern without the year-to-year amplitude shifts that
annual encoding introduces).
All active columns use **int** or **float** dtype (never ``pd.Categorical``).
``LightGBMModelStrategy.train()`` calls ``select_dtypes(exclude=['object',
'string'])``; plain numeric types survive that filter without requiring an
explicit ``categorical_feature=`` parameter on the LightGBM ``Dataset``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Calendar feature constants ─────────────────────────────────────────────

CALENDAR_FEATURE_COLS: List[str] = [
    'day_of_week_sin',
    'day_of_week_cos',
    #'day_of_year_sin',
    #'day_of_year_cos',
    'is_business_day',
]
"""Column names produced by :func:`add_calendar_features`."""


# ── Calendar helpers ───────────────────────────────────────────────────────

def compute_calendar_row(date) -> dict:
    """Return a dict of calendar feature values for a single *date*.

    Produces exactly the same values as :func:`add_calendar_features` so that
    neural schedule-mode predictions (recursive multi-step) can recompute
    features for future dates consistently.

    Parameters
    ----------
    date : datetime-like
        Any value accepted by ``pd.Timestamp()``.
    """
    ts = pd.Timestamp(date)
    return {
        'day_of_week_sin': np.sin(2 * np.pi * ts.dayofweek / 7),
        'day_of_week_cos': np.cos(2 * np.pi * ts.dayofweek / 7),
        #'day_of_year_sin': np.sin(2 * np.pi * ts.dayofyear / 365.25),
        #'day_of_year_cos': np.cos(2 * np.pi * ts.dayofyear / 365.25),
        'is_business_day': int(ts.dayofweek < 5),
    }


def add_calendar_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add calendar features to a DatetimeIndex DataFrame.

    All features use **int** or **float** dtype (never ``category``) for
    compatibility with LightGBM's ``select_dtypes`` filter.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``DatetimeIndex``.

    Returns
    -------
    (df_with_features, column_names)
        A copy of *df* with the new columns, and the list of added column
        names (matches :data:`CALENDAR_FEATURE_COLS`).
    """
    df = df.copy()
    df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    #df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    #df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['is_business_day'] = (df.index.dayofweek < 5).astype(int)
    return df, list(CALENDAR_FEATURE_COLS)


# ── Tree-model feature helpers ─────────────────────────────────────────────

def add_tree_lag_features(
    df: pd.DataFrame,
    forecast_horizon: int,
    calendar_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the full feature set for tree-based models (LightGBM / XGBoost).

    Adds in order:

    1. **Horizon lags** — ``lag_1`` … ``lag_{forecast_horizon}``
    2. **Rolling means** — 7-day and 30-day, shifted by 1 to avoid leakage
    3. **Deeper fixed lags** — 7, 14, 30 (deduplicated with horizon lags)
    4. **Calendar features** — appended if *calendar_cols* is non-empty

    The caller should follow up with ``.dropna()`` after both ``df_model``
    and ``train_df`` have been augmented (NaN rows come from lag/rolling
    warm-up).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``'y'`` column.
    forecast_horizon : int
        Determines the number of horizon-lag columns.
    calendar_cols : list[str] | None
        Column names already present in *df* to include in the feature list.
        Typically the output of :func:`add_calendar_features`.

    Returns
    -------
    (df_augmented, feature_cols)
        A copy of *df* with all new columns, and the ordered list of feature
        column names.
    """
    df = df.copy()

    # Accumulate all new columns into a dict and concat once to avoid
    # DataFrame fragmentation (PerformanceWarning from repeated inserts).
    new_cols: dict = {}

    # 1. Horizon lags
    feature_cols = [f'lag_{i}' for i in range(1, forecast_horizon + 1)]
    for i in range(1, forecast_horizon + 1):
        new_cols[f'lag_{i}'] = df['y'].shift(i)

    # 2. Rolling means — shift(1) avoids leakage (uses t-7…t-1, never t)
    shifted = df['y'].shift(1)
    new_cols['rolling_7d_mean']  = shifted.rolling(7).mean()
    new_cols['rolling_30d_mean'] = shifted.rolling(30).mean()
    feature_cols += ['rolling_7d_mean', 'rolling_30d_mean']

    # 3. Deeper fixed lags — deduplicate with horizon lags already created
    for lag_val in [7, 14, 30]:
        col_name = f'lag_{lag_val}'
        if col_name not in new_cols:
            new_cols[col_name] = df['y'].shift(lag_val)
        if col_name not in feature_cols:
            feature_cols.append(col_name)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # 4. Calendar features (already present in df if flag was set)
    if calendar_cols:
        feature_cols += calendar_cols

    return df, feature_cols


# ── Target transforms ──────────────────────────────────────────────────────

@dataclass
class TargetTransformState:
    """Holds state needed to invert forward target transforms.

    Created by :func:`apply_forward_transforms` and consumed by
    :func:`apply_inverse_transforms`.
    """
    use_log_transform: bool = False
    use_differencing: bool = False
    df_before_diff: pd.Series | None = None
    """Series in (possibly log-) space, indexed by date — used for per-date
    reconstruction of 7-day-differenced predictions."""
    use_window_norm: bool = False
    revin_stats: pd.DataFrame | None = None
    """DataFrame with columns ['mean', 'std'] indexed by date — per-date
    normalisation stats used to invert per-window normalisation."""


def apply_forward_transforms(
    df: pd.DataFrame,
    strategy_params: dict,
) -> Tuple[pd.DataFrame, TargetTransformState]:
    """Apply forward target transforms to ``df['y']`` **in place**.

    **Order**: log → window normalisation → diff.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``'y'`` column.  Modified in place; the first 7 rows
        are dropped when 7-day seasonal differencing is applied.
    strategy_params : dict
        Checked for ``use_log_transform``, ``use_differencing``,
        ``use_window_norm`` (all default ``False``), and
        ``window_norm_days`` (default ``28``).

    Returns
    -------
    (df, state)
        The (possibly shortened) DataFrame and a :class:`TargetTransformState`
        needed by :func:`apply_inverse_transforms`.
    """
    use_log = strategy_params.get('use_log_transform', False)
    use_diff = strategy_params.get('use_differencing', False)
    use_window_norm = strategy_params.get('use_window_norm', False)
    window_norm_days = int(strategy_params.get('window_norm_days', 28))

    state = TargetTransformState(
        use_log_transform=use_log,
        use_differencing=use_diff,
        use_window_norm=use_window_norm,
    )

    if use_log:
        df['y'] = np.log1p(df['y'])
        logger.info("Applied log1p transform to target.")

    if use_window_norm:
        # Compute rolling mean/std over the *preceding* window_norm_days days.
        # shift(1) prevents leakage of the current day into its own stats.
        # min_periods=window_norm_days ensures stats are only computed once a
        # full window of prior data is available — the first window_norm_days
        # rows (the look_back warm-up period) receive the identity transform
        # (mu=0, sigma=1) so they pass through unchanged and invert correctly.
        # clip(lower=1e-6) guards only against a truly constant window (std=0).
        roll = df['y'].shift(1).rolling(window=window_norm_days, min_periods=window_norm_days)
        mu = roll.mean().fillna(0.0)
        sigma = roll.std().fillna(1.0).clip(lower=1e-6)
        state.revin_stats = pd.DataFrame({'mean': mu, 'std': sigma}, index=df.index)
        df['y'] = (df['y'] - mu) / sigma
        logger.info(
            "Applied per-window normalisation (RevIN-style, window=%d days) to target.",
            window_norm_days,
        )

    if use_diff:
        # Store pre-diff values (in log / normalised space if those were applied)
        # for per-date reconstruction during inverse transform.
        state.df_before_diff = df['y'].copy()
        # Row-based 7-day diff: y[i] - y[i-7] (7th previous *row*).
        # This stays entirely within the provided data range — no calendar
        # lookup that could reach outside the index (e.g. across a COVID gap).
        # The inverse transform uses the same positional logic.
        df['y'] = df['y'].diff(7)
        # Drop the first 7 rows — diff(7) produces NaN there.
        df = df.iloc[7:]
        logger.info("Applied 7-day seasonal differencing (row-based) to target.")

    return df, state


def apply_inverse_transforms(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates_idx: pd.DatetimeIndex,
    state: TargetTransformState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Undo forward transforms on *predictions* and *actuals*.

    **Order**: undo diff first, then undo log (strict reverse of forward).

    Parameters
    ----------
    predictions, actuals : np.ndarray
        1-D arrays of equal length, in transformed space.
    dates_idx : pd.DatetimeIndex
        Corresponding dates — used for per-date diff reconstruction.
    state : TargetTransformState
        As returned by :func:`apply_forward_transforms`.

    Returns
    -------
    (predictions_original, actuals_original)
        Arrays in the original (untransformed) scale.
    """
    p, a = predictions.copy(), actuals.copy()

    if state.use_differencing and state.df_before_diff is not None:
        p_recon = np.empty_like(p)
        a_recon = np.empty_like(a)
        bdd = state.df_before_diff
        for i, date in enumerate(dates_idx[:len(p)]):
            # Mirror row-based diff(7): look up 7 rows back by position.
            # This is the exact inverse of y[pos] - y[pos-7], using only
            # rows that were present in the input data — no calendar arithmetic
            # that could reference dates outside the index (e.g. COVID gap).
            if date in bdd.index:
                pos = bdd.index.get_loc(date)
                if pos >= 7:
                    prev_val_p = bdd.iloc[pos - 7]
                    prev_val_a = prev_val_p  # same ground-truth anchor for both
                else:
                    # Fewer than 7 rows before this date in the stored series
                    # (should not occur in normal backtest; schedule-mode fallback).
                    prev_val_p = bdd.iloc[0] if i == 0 else p_recon[i - 1]
                    prev_val_a = bdd.iloc[0] if i == 0 else a_recon[i - 1]
            else:
                # Date not in training series (schedule / future mode): chain
                # from last reconstructed value, keeping p and a independent.
                prev_val_p = bdd.iloc[-1] if i == 0 else p_recon[i - 1]
                prev_val_a = bdd.iloc[-1] if i == 0 else a_recon[i - 1]
            p_recon[i] = prev_val_p + p[i]
            a_recon[i] = prev_val_a + a[i]
        p, a = p_recon, a_recon

    if state.use_window_norm and state.revin_stats is not None:
        for i, date in enumerate(dates_idx[:len(p)]):
            if date in state.revin_stats.index:
                mu = state.revin_stats.at[date, 'mean']
                sigma = state.revin_stats.at[date, 'std']
            else:
                # Fallback: use last known stats
                mu = state.revin_stats['mean'].iloc[-1]
                sigma = state.revin_stats['std'].iloc[-1]
            p[i] = p[i] * sigma + mu
            a[i] = a[i] * sigma + mu

    if state.use_log_transform:
        p = np.expm1(p)
        a = np.expm1(a)

    return p, a
