"""
Data loading and cleaning for NSE OHLCV data.

Handles:
  - 1-minute or 5-minute bar CSVs from yfinance or manual download
  - Both 'datetime' and 'date+time' column formats
  - Market hours filtering for IST (09:15 – 15:30)
"""
import pandas as pd
from pathlib import Path
from config import RAW_DIR, PROCESSED_DIR, TICKERS, MARKET_OPEN, MARKET_CLOSE
from src.utils import filter_market_hours


# ── Loading ───────────────────────────────────────────────────────────

def load_raw_csv(ticker: str, data_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Load a single ticker's OHLCV CSV into a DatetimeIndex DataFrame.
    Handles common csv column variations from yfinance and NSE data.
    """
    path = data_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data file for {ticker} at {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Build datetime index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    elif "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        # Try first column
        df["datetime"] = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    # Drop rows where datetime parsing failed
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    # Standardise column names
    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "open":
            rename_map[col] = "open"
        elif cl == "high":
            rename_map[col] = "high"
        elif cl == "low":
            rename_map[col] = "low"
        elif cl == "close":
            rename_map[col] = "close"
        elif cl == "volume":
            rename_map[col] = "volume"
    df = df.rename(columns=rename_map)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}.csv is missing columns: {missing}")

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[~df.index.duplicated(keep="first")]
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(float)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any remaining NaNs in price columns
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df


def load_all_tickers(
    tickers: list[str] = TICKERS,
    data_dir: Path = RAW_DIR,
    market_hours: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load all tickers, optionally filtering to market hours."""
    data = {}
    for ticker in tickers:
        try:
            df = load_raw_csv(ticker, data_dir)
            if market_hours and hasattr(df.index, 'time'):
                df = filter_market_hours(df, MARKET_OPEN, MARKET_CLOSE)
            if len(df) > 0:
                data[ticker] = df
        except Exception as e:
            print(f"Warning: Could not load {ticker}: {e}")
    return data


# ── Alignment ─────────────────────────────────────────────────────────

def align_tickers(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Inner-join all tickers on their datetime index.
    Returns a wide DataFrame with MultiLevel columns: (ticker, ohlcv).
    """
    aligned = pd.concat(
        {ticker: df for ticker, df in data.items()},
        axis=1,
    )
    aligned = aligned.dropna()  # keep only bars where ALL tickers have data
    return aligned


def save_processed(panel: pd.DataFrame, name: str = "panel") -> Path:
    """Save aligned panel to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    panel.to_parquet(path)
    return path


def load_processed(name: str = "panel") -> pd.DataFrame:
    """Load previously saved panel."""
    path = PROCESSED_DIR / f"{name}.parquet"
    return pd.read_parquet(path)
