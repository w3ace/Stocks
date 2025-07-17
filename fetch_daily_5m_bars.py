import yfinance as yf
import pandas as pd
import os
import logging
import datetime
import sys

from stock_functions import round_numeric_cols

# Set up logging
logging.basicConfig(filename='stock_info.log', level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')

def get_date_str(date=None):
    if date:
        return date.strftime('%Y%m%d')
    else:
        return datetime.datetime.now().strftime('%Y%m%d')

def fetch_5min_data(ticker, date_str):
    try:
        first_letter = ticker[0].upper()
        directory = f"Datasets/Ticker/Daily/{date_str}/{first_letter}/{ticker}"
        os.makedirs(directory, exist_ok=True)
        filename = f"{directory}/{ticker}.bars"

        if not os.path.exists(filename):
            stock = yf.Ticker(ticker)
            data = stock.history(interval='5m', start='2023-01-01', end='2023-12-31')  # Adjust the start and end dates as needed
            data = round_numeric_cols(data)
            data.to_csv(filename)
            print(f"Fetched and saved 5-minute interval data for {ticker}")
        else:
            print(f"Data file for {ticker} already exists. Skipping...")
    except Exception as e:
        logging.error(f"Error fetching 5-minute data for {ticker}: {e}")

def main(date=None):
    df = pd.read_csv('Datasets/us_stocks.csv')
    tickers = df['symbol'].tolist()

    date_str = get_date_str(date)

    for ticker in tickers:
        fetch_5min_data(ticker, date_str)

if __name__ == "__main__":
    date = None
    if len(sys.argv) > 1:
        try:
            date = datetime.datetime.strptime(sys.argv[1], '%Y%m%d')
        except ValueError:
            print(f"Invalid date format: {sys.argv[1]}. Use YYYYMMDD.")
            sys.exit(1)
    main(date)
