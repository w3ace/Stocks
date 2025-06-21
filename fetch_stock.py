import logging
import yfinance as yf

logging.basicConfig(level=logging.INFO)

def fetch_stock(symbol, start_date=0, end_date=0, period="1mo"):
    try:
        if start_date and end_date:
            hourly_data = yf.download(
                symbol, interval="1h", start=start_date, end=end_date
            )
            daily_data = yf.download(
                symbol, interval="1d", start=start_date, end=end_date
            )
        else:
            hourly_data = yf.download(symbol, interval="1h", period=period)
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
