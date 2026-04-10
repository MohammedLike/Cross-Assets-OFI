"""
Generate synthetic NSE 1-minute OHLCV data for pipeline testing.

This creates realistic-looking (but fake) data so the full pipeline
can be tested end-to-end without real market data. The synthetic data
mimics NSE characteristics: market hours 09:15–15:30 IST, weekdays only,
correlated price movements across assets.

Usage:
    python scripts/generate_sample_data.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import TICKERS, RAW_DIR

SEED = 42
MONTHS = 12  # generates ~12 months of data
BARS_PER_DAY = 375  # 09:15 to 15:30 = 375 minutes


def generate_market_dates(start: str = "2024-01-02", months: int = MONTHS) -> list[pd.Timestamp]:
    """Generate weekday trading dates."""
    end = pd.Timestamp(start) + pd.DateOffset(months=months)
    dates = pd.bdate_range(start, end)
    return list(dates)


def generate_intraday_index(dates: list[pd.Timestamp]) -> pd.DatetimeIndex:
    """Create 1-minute bar timestamps for all trading days."""
    timestamps = []
    for date in dates:
        market_open = date.replace(hour=9, minute=15, second=0)
        bars = pd.date_range(market_open, periods=BARS_PER_DAY, freq="1min")
        timestamps.extend(bars)
    return pd.DatetimeIndex(timestamps)


def generate_ohlcv(
    index: pd.DatetimeIndex,
    base_price: float = 100.0,
    volatility: float = 0.001,
    mean_volume: float = 10000.0,
    common_factor: np.ndarray = None,
    factor_loading: float = 0.5,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with optional common factor exposure.
    """
    n = len(index)
    if rng is None:
        rng = np.random.default_rng(SEED)

    # Generate returns with common factor + idiosyncratic
    idio_returns = rng.normal(0, volatility, n)
    if common_factor is not None:
        returns = factor_loading * common_factor + (1 - factor_loading) * idio_returns
    else:
        returns = idio_returns

    # Reset at each day boundary to prevent extreme drift
    dates = index.date
    close = np.zeros(n)
    close[0] = base_price
    prev_date = dates[0]
    day_open = base_price

    for i in range(1, n):
        if dates[i] != prev_date:
            # New day: small overnight gap
            day_open = close[i - 1] * (1 + rng.normal(0, 0.002))
            close[i] = day_open * (1 + returns[i])
            prev_date = dates[i]
        else:
            close[i] = close[i - 1] * (1 + returns[i])

    # Generate OHLV from close
    spread = np.abs(rng.normal(0, volatility * 0.5, n))
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = close * (1 + rng.normal(0, volatility * 0.3, n))

    # Ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Volume: U-shaped intraday pattern
    bar_of_day = np.tile(
        np.concatenate([
            np.linspace(2, 0.5, BARS_PER_DAY // 3),
            np.ones(BARS_PER_DAY // 3) * 0.5,
            np.linspace(0.5, 1.5, BARS_PER_DAY - 2 * (BARS_PER_DAY // 3)),
        ]),
        len(set(dates)),
    )[:n]
    volume = rng.poisson(mean_volume * bar_of_day)

    return pd.DataFrame({
        "datetime": index,
        "open": np.round(open_, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": volume,
    })


def main():
    rng = np.random.default_rng(SEED)

    dates = generate_market_dates()
    index = generate_intraday_index(dates)
    n = len(index)

    # Common market factor (drives cross-asset correlation)
    common_factor = rng.normal(0, 0.001, n)

    # Asset-specific parameters
    params = {
        "NIFTY":     {"base_price": 22000, "volatility": 0.0008, "mean_volume": 50000, "factor_loading": 0.7},
        "BANKNIFTY": {"base_price": 47000, "volatility": 0.0010, "mean_volume": 40000, "factor_loading": 0.65},
        "HDFCBANK":  {"base_price": 1600,  "volatility": 0.0012, "mean_volume": 20000, "factor_loading": 0.5},
        "RELIANCE":  {"base_price": 2500,  "volatility": 0.0011, "mean_volume": 25000, "factor_loading": 0.45},
        "INFY":      {"base_price": 1500,  "volatility": 0.0013, "mean_volume": 15000, "factor_loading": 0.4},
    }

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        p = params[ticker]
        df = generate_ohlcv(
            index,
            base_price=p["base_price"],
            volatility=p["volatility"],
            mean_volume=p["mean_volume"],
            common_factor=common_factor,
            factor_loading=p["factor_loading"],
            rng=np.random.default_rng(SEED + hash(ticker) % 1000),
        )
        path = RAW_DIR / f"{ticker}.csv"
        df.to_csv(path, index=False)
        print(f"  {ticker:12s}  {len(df):>8,} bars  -> {path}")

    print(f"\nGenerated {len(TICKERS)} synthetic datasets in {RAW_DIR}")
    print("Run notebooks/01_data_exploration.ipynb to validate and process.")


if __name__ == "__main__":
    main()
