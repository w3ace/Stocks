import logging
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO)

def fetch_stock(symbol, start_date=0, end_date=0, period="1mo", interval="1h"):
    """Return intraday and daily data for ``symbol`` using *yfinance*.

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

        return data

    except Exception as e:
        logging.warning(f"Failed to download data for {symbol}: {e}")
        return None
