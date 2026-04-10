"""
Feature matrix construction for the cross-asset OFI study.

Two feature sets per target asset:
  1. own-only  : target asset's OFI at all horizons
  2. full      : own OFI + signal asset (NIFTY) OFI at all horizons

The incremental R² test compares full vs own-only.
"""
import numpy as np
import pandas as pd
from config import OFI_HORIZONS, SIGNAL_ASSET, DEFAULT_FWD_HORIZON
from src.utils import forward_log_return, winsorise


def build_target(
    close: pd.Series,
    horizon: int = DEFAULT_FWD_HORIZON,
) -> pd.Series:
    """Forward log-return of the target asset."""
    return forward_log_return(close, horizon)


def _ofi_columns(ticker: str, horizons: list[int] = OFI_HORIZONS) -> list[str]:
    """Column names for a ticker's OFI features."""
    return [f"{ticker}_ofi_{h}" for h in horizons]


def build_own_features(
    ofi_df: pd.DataFrame,
    target_ticker: str,
    horizons: list[int] = OFI_HORIZONS,
) -> pd.DataFrame:
    """Return only the target asset's own OFI columns."""
    cols = _ofi_columns(target_ticker, horizons)
    return ofi_df[cols].copy()


def build_full_features(
    ofi_df: pd.DataFrame,
    target_ticker: str,
    signal_ticker: str = SIGNAL_ASSET,
    horizons: list[int] = OFI_HORIZONS,
) -> pd.DataFrame:
    """Return own-asset + cross-asset (signal) OFI columns."""
    own_cols = _ofi_columns(target_ticker, horizons)
    cross_cols = _ofi_columns(signal_ticker, horizons)
    # Avoid duplicates if target == signal
    all_cols = list(dict.fromkeys(own_cols + cross_cols))
    return ofi_df[all_cols].copy()


def prepare_dataset(
    ofi_df: pd.DataFrame,
    close: pd.Series,
    target_ticker: str,
    feature_set: str = "full",
    fwd_horizon: int = DEFAULT_FWD_HORIZON,
    winsorise_sigma: float = 5.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build aligned (X, y) ready for modelling.

    Parameters
    ----------
    ofi_df : output of compute_all_ofi
    close : close price series for the target asset
    target_ticker : e.g. 'BANKNIFTY'
    feature_set : 'own' or 'full'
    fwd_horizon : forward return horizon in minutes
    winsorise_sigma : outlier clipping threshold

    Returns
    -------
    X, y : aligned feature matrix and target, NaN rows dropped.
    """
    y = build_target(close, fwd_horizon)
    y = winsorise(y, winsorise_sigma)
    y.name = "fwd_return"

    if feature_set == "own":
        X = build_own_features(ofi_df, target_ticker)
    else:
        X = build_full_features(ofi_df, target_ticker)

    # Align and drop NaNs
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[X.columns]
    y = combined["fwd_return"]

    return X, y
