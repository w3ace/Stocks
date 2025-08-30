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
    # yfinance rejects supplying ``period`` alongside ``start``/``end``.  If a
    # date range is given we drop the ``period`` argument to avoid empty
    # downloads.  Cached filenames mirror the arguments actually used.
    use_period = period if not (start or end) else None
    cache_key = f"{ticker}_{start}_{end}_{use_period}.csv".replace("None", "-")
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists() and not force:
        try:
            return pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            cache_path.unlink(missing_ok=True)

    kwargs = {"auto_adjust": True, "progress": False}
    if start or end:
        kwargs.update({"start": start, "end": end})
    else:
        kwargs.update({"period": period})

    df = yf.download(ticker, **kwargs)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    df.index.name = "Date"
    df.to_csv(cache_path)
    return df
