"""
Downloads the AAPL OHLC data used in Section 1 (returns distribution / QQ-plots).

Run once before `returns_analysis.py`:
    python src/fetch_data.py
"""
import os

import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

START = "2020-08-01"
END = "2023-10-25"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    daily = _flatten(yf.download("AAPL", start=START, end=END, interval="1d", progress=False))
    daily.to_csv(os.path.join(DATA_DIR, "daily_aapl.csv"), index=False)
    print(f"Saved {len(daily)} daily rows to data/daily_aapl.csv")

    weekly = _flatten(yf.download("AAPL", start=START, end=END, interval="1wk", progress=False))
    weekly.to_csv(os.path.join(DATA_DIR, "weekly_aapl.csv"), index=False)
    print(f"Saved {len(weekly)} weekly rows to data/weekly_aapl.csv")


if __name__ == "__main__":
    main()
