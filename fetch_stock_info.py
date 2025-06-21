import yfinance as yf
import pandas as pd
import os
import time
import logging

# Set up logging
logging.basicConfig(filename='stock_info.log', level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')

def get_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get('marketCap', 0)
        return market_cap
    except Exception as e:
        logging.error(f"Error fetching market cap for {ticker}: {e}")
        return 0

def get_days_to_check(market_cap):
    if market_cap < 100_000_000:
        return 4
    elif market_cap < 250_000_000:
        return 3
    elif market_cap < 1_000_000_000:
        return 2
    elif market_cap > 10_000_000_000:
        return 1
    return 5

def check_and_fetch_info(ticker, file_mod_times):
    print(f"Working on {ticker}")
    market_cap = get_market_cap(ticker)
    days_to_check = get_days_to_check(market_cap)

    if isinstance(ticker, str):
        first_letter = ticker[0].upper()
    else:
        print(f"Unexpected type for ticker: {type(ticker)}")
        return

    directory = f"Datasets/Ticker/{first_letter}"
    filename = f"{directory}/{ticker}.info"

    # os.makedirs(directory, exist_ok=True)

    current_time = time.time()

    if filename in file_mod_times:
        file_mod_time = file_mod_times[filename]
        file_age_days = (current_time - file_mod_time) / (60 * 60 * 24)

        if file_age_days <= days_to_check:
            print(f"Info file for {ticker} is up-to-date (less than {days_to_check} days old). Skipping...")
            return

    fetch_info(ticker)

def fetch_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        first_letter = ticker[0].upper()
        directory = f"Datasets/Ticker/{first_letter}"
        filename = f"{directory}/{ticker}.info"

        with open(filename, 'w') as f:
            f.write(str(info))

        print(f"Fetched and saved info for {ticker}")
        time.sleep(0.2)
    except Exception as e:
        logging.error(f"Error fetching info for {ticker}: {e}")

def get_file_mod_times(base_directory):
    file_mod_times = {}
    for root, _, files in os.walk(base_directory):
        for file in files:
            filepath = os.path.join(root, file)
            file_mod_times[filepath.replace("\\","/")] = os.path.getmtime(filepath)
    return file_mod_times

def main():
    df = pd.read_csv('Datasets/us_stocks.csv')
    tickers = df['symbol'].tolist()

    base_directory = 'Datasets/Ticker'
    file_mod_times = get_file_mod_times(base_directory)

    for ticker in tickers:
        check_and_fetch_info(ticker, file_mod_times)

if __name__ == "__main__":
    main()
