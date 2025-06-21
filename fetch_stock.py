import logging
import yfinance as yf

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
            hourly_data = yf.download(
                symbol, interval=interval, start=start_date, end=end_date
            )
            daily_data = yf.download(
                symbol, interval="1d", start=start_date, end=end_date
            )
        else:
            hourly_data = yf.download(symbol, interval=interval, period=period)
            daily_data = yf.download(symbol, interval="1d", period=period)

        if hourly_data.empty or daily_data.empty:
            logging.warning(f"No data returned for {symbol}")
            return None, None

        hourly_data.reset_index(inplace=True)
        daily_data.reset_index(inplace=True)

        return hourly_data, daily_data

    except Exception as e:
        logging.warning(f"Failed to download data for {symbol}: {e}")
        return None, None
