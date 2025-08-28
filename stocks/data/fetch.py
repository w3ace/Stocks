from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ticker(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download OHLC data for *ticker* using yfinance with on-disk caching."""
    cache_key = f"{ticker}_{start}_{end}_{period}.csv".replace("None", "-")
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        period=period,
        auto_adjust=True,
        progress=False,
    )
    df.to_csv(cache_path)
    return df
