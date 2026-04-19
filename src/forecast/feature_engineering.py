"""
Feature engineering and target transforms for the forecast pipeline.

This module centralises all feature-construction logic that is shared across
model strategies, keeping the pipeline orchestrator (``tfm_pipeline``) and the
individual strategy files free of feature-level implementation details.

Three orthogonal capabilities — each controlled by a boolean flag in
``strategy_params`` — can be combined freely, yielding up to 8 experiment
configurations:

+---------------------+------------------------------------------------------+
| Flag                | Effect                                               |
+=====================+======================================================+
| use_log_transform   | ``np.log1p`` / ``np.expm1`` round-trip on target     |
+---------------------+------------------------------------------------------+
| use_differencing    | First-order differencing / per-date reconstruction   |
+---------------------+------------------------------------------------------+
| use_calendar_features | Day-of-week, month, sin/cos day-of-year, business  |
|                     | day indicator — added to the DataFrame               |
+---------------------+------------------------------------------------------+

**Ordering rules**

* Forward transforms:  log **first** → diff **second**
  (diff of log-values gives log-returns — a well-known stationary transform).
* Inverse transforms:  undo diff **first** → undo log **second**
  (strict reverse of the forward order; swapping corrupts the result).

**Why per-date reconstruction instead of cumsum?**

Cumulative sum propagates prediction error: if prediction 1 is off by +100
every subsequent prediction inherits that bias.  Per-date lookup uses the
*actual* previous-day value (from ``df_before_diff``), keeping each
prediction's error independent.  The cumsum fallback is only used in
schedule mode where actual previous values are unavailable.

**Calendar features — dtype choice**

All calendar columns use **int** or **float** dtype (never ``pd.Categorical``).
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
    'day_of_week',
    'month',
    'day_of_year_sin',
    'day_of_year_cos',
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
        'day_of_week': ts.dayofweek,
        'month': ts.month,
        'day_of_year_sin': np.sin(2 * np.pi * ts.dayofyear / 365.25),
        'day_of_year_cos': np.cos(2 * np.pi * ts.dayofyear / 365.25),
        'is_business_day': int(pd.tseries.offsets.BDay().is_on_offset(ts)),
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
    df['day_of_week']     = df.index.dayofweek
    df['month']           = df.index.month
    df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['is_business_day'] = df.index.map(
        lambda x: int(pd.tseries.offsets.BDay().is_on_offset(x))
    )
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

    # 1. Horizon lags
    feature_cols = [f'lag_{i}' for i in range(1, forecast_horizon + 1)]
    for i in range(1, forecast_horizon + 1):
        df[f'lag_{i}'] = df['y'].shift(i)

    # 2. Rolling means — shift(1) avoids leakage (uses t-7…t-1, never t)
    df['rolling_7d_mean']  = df['y'].shift(1).rolling(7).mean()
    df['rolling_30d_mean'] = df['y'].shift(1).rolling(30).mean()
    feature_cols += ['rolling_7d_mean', 'rolling_30d_mean']

    # 3. Deeper fixed lags — deduplicate with horizon lags already created
    for lag_val in [7, 14, 30]:
        col_name = f'lag_{lag_val}'
        if col_name not in df.columns:
            df[col_name] = df['y'].shift(lag_val)
        if col_name not in feature_cols:
            feature_cols.append(col_name)

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
    reconstruction of differenced predictions."""


def apply_forward_transforms(
    df: pd.DataFrame,
    strategy_params: dict,
) -> Tuple[pd.DataFrame, TargetTransformState]:
    """Apply forward target transforms to ``df['y']`` **in place**.

    **Order**: log first, then diff (diff of log-values = log-returns).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``'y'`` column.  Modified in place; rows may be dropped
        when differencing removes the first NaN.
    strategy_params : dict
        Checked for ``use_log_transform`` and ``use_differencing`` (both
        default ``False``).

    Returns
    -------
    (df, state)
        The (possibly shortened) DataFrame and a :class:`TargetTransformState`
        needed by :func:`apply_inverse_transforms`.
    """
    use_log = strategy_params.get('use_log_transform', False)
    use_diff = strategy_params.get('use_differencing', False)

    state = TargetTransformState(use_log_transform=use_log, use_differencing=use_diff)

    if use_log:
        df['y'] = np.log1p(df['y'])
        logger.info("Applied log1p transform to target.")

    if use_diff:
        # Store pre-diff values (in log space if log was applied) for
        # per-date reconstruction during inverse transform.
        state.df_before_diff = df['y'].copy()
        df['y'] = df['y'].diff()
        # Drop the first row — diff() produces NaN there; keeping it would
        # propagate NaN into every downstream lag/rolling feature.
        df = df.iloc[1:]
        logger.info("Applied first-order differencing to target.")

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
        for i, date in enumerate(dates_idx[:len(p)]):
            prev_date = date - pd.Timedelta(days=1)
            if prev_date in state.df_before_diff.index:
                prev_val = state.df_before_diff.loc[prev_date]
            else:
                # Fallback for schedule mode (future dates beyond training data):
                # use cumulative sum from last known value.
                prev_val = (
                    state.df_before_diff.iloc[-1] if i == 0 else p_recon[i - 1]
                )
            p_recon[i] = prev_val + p[i]
            a_recon[i] = prev_val + a[i]
        p, a = p_recon, a_recon

    if state.use_log_transform:
        p = np.expm1(p)
        a = np.expm1(a)

    return p, a
