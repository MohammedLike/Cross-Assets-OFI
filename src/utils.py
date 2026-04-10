"""
Utility helpers for the Cross-Asset OFI project.
"""
import numpy as np
import pandas as pd


def winsorise(series: pd.Series, n_sigma: float = 5) -> pd.Series:
    """Clip values beyond n_sigma standard deviations from the mean."""
    mu, sigma = series.mean(), series.std()
    lower, upper = mu - n_sigma * sigma, mu + n_sigma * sigma
    return series.clip(lower, upper)


def filter_market_hours(
    df: pd.DataFrame,
    open_time: str = "09:15",
    close_time: str = "15:30",
) -> pd.DataFrame:
    """Keep only rows within NSE market hours."""
    t = df.index.time
    start = pd.Timestamp(open_time).time()
    end = pd.Timestamp(close_time).time()
    return df.loc[(t >= start) & (t <= end)]


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Compute forward log-return: log(close_{t+h} / close_t)."""
    return np.log(close.shift(-horizon) / close)


def lag_series(series: pd.Series, n_lags: int) -> pd.DataFrame:
    """Create a DataFrame with lagged copies of a series."""
    return pd.concat(
        {f"lag_{i}": series.shift(i) for i in range(1, n_lags + 1)},
        axis=1,
    )
