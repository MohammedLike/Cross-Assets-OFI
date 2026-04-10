"""
Fetch real NSE stock data via yfinance for the Cross-Asset OFI study.

Downloads 1-minute OHLCV data for the configured tickers using the
maximum available history from yfinance (≈7 days for 1m, or longer
intervals for longer history).

For intraday OFI analysis, we use 5-minute bars from Yahoo Finance
(maximum ~60 days history) and also fetch 1-day bars for longer-term
regime analysis.

Usage:
    python scripts/fetch_real_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf
from config import RAW_DIR, TICKERS

# Yahoo Finance ticker mapping for NSE stocks
NSE_TICKER_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "HDFCBANK": "HDFCBANK.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
}

# Use 5-minute bars to get ~60 days of data (best tradeoff)
INTERVAL = "5m"
PERIOD = "60d"


def fetch_ticker(ticker: str, yahoo_symbol: str) -> pd.DataFrame:
    """Download data for a single ticker and format it."""
    print(f"  Fetching {ticker} ({yahoo_symbol})...")
    
    obj = yf.Ticker(yahoo_symbol)
    df = obj.history(period=PERIOD, interval=INTERVAL)
    
    if df.empty:
        print(f"  ⚠ No data returned for {ticker}. Trying daily fallback...")
        df = obj.history(period="1y", interval="1d")
    
    if df.empty:
        raise ValueError(f"No data available for {ticker} ({yahoo_symbol})")
    
    # Standardise columns
    df = df.reset_index()
    
    # Handle different index column names
    date_col = None
    for col in df.columns:
        if "date" in col.lower() or "datetime" in col.lower():
            date_col = col
            break
    
    if date_col is None:
        date_col = df.columns[0]
    
    df = df.rename(columns={date_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Ensure we have the right columns
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "open": col_map[col] = "open"
        elif cl == "high": col_map[col] = "high"
        elif cl == "low": col_map[col] = "low"
        elif cl == "close": col_map[col] = "close"
        elif cl == "volume": col_map[col] = "volume"
    
    df = df.rename(columns=col_map)
    
    # Filter to required columns
    required = ["datetime", "open", "high", "low", "close", "volume"]
    available = [c for c in required if c in df.columns]
    df = df[available]
    
    # Remove timezone info for consistency
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    
    print(f"    ✓ {len(df)} bars from {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def main():
    print("=" * 60)
    print("Cross-Asset OFI — Real Data Fetcher")
    print("=" * 60)
    print(f"\nInterval: {INTERVAL}, Period: {PERIOD}")
    print(f"Tickers: {', '.join(TICKERS)}\n")
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    for ticker in TICKERS:
        yahoo_sym = NSE_TICKER_MAP.get(ticker, f"{ticker}.NS")
        try:
            df = fetch_ticker(ticker, yahoo_sym)
            path = RAW_DIR / f"{ticker}.csv"
            df.to_csv(path, index=False)
            print(f"    Saved to {path}\n")
        except Exception as e:
            print(f"    ✗ Error: {e}\n")
    
    print("Done! Data ready in", RAW_DIR)


if __name__ == "__main__":
    main()
