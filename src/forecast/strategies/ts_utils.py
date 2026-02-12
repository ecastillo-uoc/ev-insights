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
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val
