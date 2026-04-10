"""
Order Flow Imbalance (OFI) computation from 1-minute OHLCV data.

Since we lack Level-2 order book data, we use the tick rule:
  direction = sign(close_t - close_{t-1})
  buy_vol  = volume if direction > 0, else 0
  sell_vol = volume if direction < 0, else 0
  OFI_h    = (rolling_buy_h - rolling_sell_h) / (rolling_buy_h + rolling_sell_h)

This normalised OFI lies in [-1, +1] and is comparable across assets.
"""
import numpy as np
import pandas as pd
from config import OFI_HORIZONS


def sign_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the tick rule to classify each bar's volume as buy or sell.

    Parameters
    ----------
    df : DataFrame with columns ['close', 'volume'] and a DatetimeIndex.

    Returns
    -------
    DataFrame with additional columns: 'direction', 'buy_vol', 'sell_vol'.
    """
    out = df.copy()

    # Price direction: +1, -1, or 0
    price_diff = out["close"].diff()
    direction = np.sign(price_diff)

    # Propagate last non-zero direction through zeros
    direction = direction.replace(0, np.nan).ffill().fillna(0)
    out["direction"] = direction

    out["buy_vol"] = np.where(direction > 0, out["volume"], 0.0)
    out["sell_vol"] = np.where(direction < 0, out["volume"], 0.0)

    return out


def compute_ofi(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Compute normalised OFI over a trailing window of `horizon` minutes.

    Parameters
    ----------
    df : DataFrame that already has 'buy_vol' and 'sell_vol' columns
         (output of sign_volume).
    horizon : int, rolling window size in bars (minutes).

    Returns
    -------
    pd.Series of OFI values in [-1, +1].
    """
    roll_buy = df["buy_vol"].rolling(horizon, min_periods=horizon).sum()
    roll_sell = df["sell_vol"].rolling(horizon, min_periods=horizon).sum()
    total = roll_buy + roll_sell
    ofi = (roll_buy - roll_sell) / total
    ofi = ofi.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return ofi


def compute_all_ofi(
    data: dict[str, pd.DataFrame],
    horizons: list[int] = OFI_HORIZONS,
) -> pd.DataFrame:
    """
    Compute OFI at all horizons for every ticker.

    Parameters
    ----------
    data : dict mapping ticker -> DataFrame (raw OHLCV, already aligned).
    horizons : list of horizon sizes in minutes.

    Returns
    -------
    DataFrame with columns like 'NIFTY_ofi_1', 'NIFTY_ofi_5', etc.
    """
    ofi_frames = {}

    for ticker, df in data.items():
        signed = sign_volume(df)
        for h in horizons:
            col_name = f"{ticker}_ofi_{h}"
            ofi_frames[col_name] = compute_ofi(signed, h)

    ofi_df = pd.DataFrame(ofi_frames)
    return ofi_df
