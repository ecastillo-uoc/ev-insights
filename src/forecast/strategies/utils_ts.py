"""Time-series feature engineering helpers.

Provides lag-feature construction, cyclic time features, and the symmetric
mean absolute percentage error (sMAPE) metric.
"""
import numpy as np
import pandas as pd

def add_lags(df_, lag_col, lags, lag_windows=None):
    """Returns a new dataframe that also includes lagged values of the lag_col column of df.
       For long lags (1 day or more), also include mean, min, max over lag_window."""

    df = df_.copy()
    # Add lagged values
    for lag, lag_str in lags.items():
        df["lag_" + lag_str] = df[lag_col].shift(int(lag))
    if lag_windows is not None:
        # Add mean/min/max over lags windows
        for lag_w, lag_w_str in lag_windows.items():
            for lag, lag_str in lags.items():
                df["lag_" + lag_str + "mean" + lag_w_str] = df[lag_col].shift(
                    int(lag) - int(lag_w) // 2).rolling(int(lag_w) + 1).mean()
                df["lag_" + lag_str + "max" + lag_w_str] = df[lag_col].shift(
                    int(lag) - int(lag_w) // 2).rolling(int(lag_w) + 1).max()
                df["lag_" + lag_str + "min" + lag_w_str] = df[lag_col].shift(
                    int(lag) - int(lag_w) // 2).rolling(int(lag_w) + 1).min()

    return df

def add_timefeat_df(df_):
    """Add time-related features to df_"""

    # Create time features (for periodic features: use sin and cos)
    df = df_.copy()
    df["day_of_week"] = df.index.dayofweek.astype("category")
    df["day_of_year_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df["month"] = df.index.month.astype("category")
    df['is_business_day'] = df.index.map(lambda x: pd.tseries.offsets.BDay().is_on_offset(x)).astype("category")

    return df

def smape(preds, target):
    """Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    preds : array-like
        Predicted values.
    target : array-like
        Actual (ground-truth) values.

    Returns
    -------
    float
        sMAPE value in the range [0, 200].  Pairs where both
        ``preds`` and ``target`` are zero are excluded.
    """
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    n = masked_arr.sum()  # exclude (0,0) pairs from denominator too
    if n == 0:
        return float('nan')
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def mase(preds, target, train_actuals, m: int = 1):
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Scales the MAE of the forecast by the MAE of the naïve lag-*m* predictor
    computed on the *training* series.  A MASE < 1 means the model beats the
    naïve baseline; MASE = 1 is equivalent to the naïve baseline.

    Parameters
    ----------
    preds : array-like
        Predicted values (original scale, already inverse-transformed).
    target : array-like
        Actual values (original scale).
    train_actuals : array-like
        Training-set actuals in original scale.  Used only for the naïve
        denominator; must have at least ``m + 1`` elements.
    m : int
        Seasonal period for the naïve baseline.  Default 1 (random-walk /
        yesterday's value).

    Returns
    -------
    float
        MASE value.  Returns NaN when the naïve denominator is zero (flat
        training series) or when fewer than ``m + 1`` training points exist.
    """
    preds = np.asarray(preds, dtype=float)
    target = np.asarray(target, dtype=float)
    train_actuals = np.asarray(train_actuals, dtype=float)

    if len(train_actuals) <= m:
        return float('nan')

    naive_errors = np.abs(train_actuals[m:] - train_actuals[:-m])
    scale = np.mean(naive_errors)
    if scale == 0:
        return float('nan')

    mae = np.mean(np.abs(preds - target))
    return mae / scale
