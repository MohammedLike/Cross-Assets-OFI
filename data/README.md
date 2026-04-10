# Data Sourcing

## Required Data

1-minute OHLCV bars for these NSE instruments:
- **NIFTY** (Nifty 50 Futures)
- **BANKNIFTY** (Bank Nifty Futures)
- **HDFCBANK** (HDFC Bank)
- **RELIANCE** (Reliance Industries)
- **INFY** (Infosys)

## Where to Get It

### Option 1 — Kaggle (recommended)
Search Kaggle for "NSE 1 minute data" or "NSE intraday data". Common datasets:
- NSE Stocks Intraday Data
- Indian Stock Market 1-min OHLCV

Download CSVs and rename them as: `NIFTY.csv`, `BANKNIFTY.csv`, `HDFCBANK.csv`, `RELIANCE.csv`, `INFY.csv`.

### Option 2 — Yahoo Finance via yfinance
```python
import yfinance as yf
# Note: Yahoo only provides 1-min data for the last 7 days for Indian tickers.
# For longer history, 5-min or 15-min data is available for 60 days.
df = yf.download("^NSEI", interval="1m", period="7d")
```

### Option 3 — NSE/BSE data vendors
Paid providers like Truedata, Global Datafeeds, or Kite (Zerodha) offer historical intraday data.

## Expected CSV Format

Each CSV should have these columns (case-insensitive):

| Column   | Description                    |
|----------|--------------------------------|
| datetime | or separate `date` + `time`    |
| open     | Opening price of the bar       |
| high     | High price                     |
| low      | Low price                      |
| close    | Closing price                  |
| volume   | Total volume in the bar        |

Place all CSVs in `data/raw/`.

## Processing

Run `notebooks/01_data_exploration.ipynb` to:
1. Load and validate the raw data
2. Filter to NSE market hours (09:15–15:30 IST)
3. Align all tickers and save as `data/processed/panel.parquet`
