import logging
import hashlib
from pathlib import Path

import time

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)

# Directory used for caching yfinance responses
CACHE_DIR = Path(__file__).resolve().parent / "yfinance_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(symbol, start_date, end_date, period, interval):
    """Return the cache file path for a given set of yfinance arguments."""
    key = f"{symbol}_{start_date}_{end_date}_{period}_{interval}"
    h = hashlib.md5(key.encode()).hexdigest()
    sub = CACHE_DIR / h[:2]
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{h}.pkl"


def fetch_stock(symbol, start_date=0, end_date=0, period="1mo", interval="1h"):
    """Return intraday and daily data for ``symbol`` using *yfinance*.

    Results are cached to avoid repeated downloads. When requesting data for
    the current trading day, caching is disabled until after the US/Eastern
    market close (4pm) to ensure partial intraday data is not stored.

    Parameters
    ----------
    symbol : str
        Ticker symbol to download.
    start_date, end_date : datetime-like, optional
        If provided, data is fetched between these dates.  When omitted,
        ``period`` is used instead.
    period : str, default ``"1mo"``
        Period string understood by :func:`yfinance.download`.
    interval : str, default ``"1h"``
        Interval for intraday data. Daily data is always downloaded at ``1d``.
    """
    try:
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date)
                now = pd.Timestamp.now(tz=end_dt.tzinfo) if end_dt.tzinfo else pd.Timestamp.now()

                if end_dt.normalize() == now.normalize():
                    now_est = now.tz_convert("US/Eastern") if now.tzinfo else now.tz_localize("UTC").tz_convert("US/Eastern")
                    four_pm_est = now_est.normalize() + pd.Timedelta(hours=16)
                    end_date = four_pm_est if now_est >= four_pm_est else now - pd.Timedelta(minutes=5)
            except Exception:
                end_dt = None
        else:
            end_dt = None

#        time.sleep(0.1)
        cache_file = _cache_path(symbol, start_date, end_date, period, interval)

        # Cache control
        cache_enabled = True
        now_est = pd.Timestamp.now(tz="US/Eastern")
        four_pm = pd.Timestamp("16:00", tz="US/Eastern").time()

        if end_dt:
            try:
                if end_dt.tzinfo is None:
                    end_dt = end_dt.tz_localize("UTC")
                end_dt_est = end_dt.tz_convert("US/Eastern")

                if end_dt_est.normalize() == now_est.normalize() and now_est.time() < four_pm:
                    cache_enabled = False
            except Exception:
                pass
        else:
            if now_est.time() < four_pm:
                cache_enabled = False

        if cache_enabled and cache_file.exists():
            try:
                data = pd.read_pickle(cache_file)
                return data
            except Exception as e:
                logging.warning(f"Failed to read cache {cache_file}: {e}")

        if start_date and end_date:
            data = yf.download(
                symbol, interval=interval, start=start_date, end=end_date, auto_adjust=True
            )
        else:
            data = yf.download(symbol, interval=interval, period=period, auto_adjust=True)

        if data.empty:
            logging.warning(f"No data returned for {symbol}")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.reset_index(inplace=True)

        if cache_enabled:
            try:
                data.to_pickle(cache_file)
            except Exception as e:
                logging.warning(f"Failed to write cache {cache_file}: {e}")

        return data

    except Exception as e:
        logging.warning(f"Failed to download data for {symbol}: {e}")
        return None
