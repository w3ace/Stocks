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
    if market_cap < 1_000_000:
        return 5
    elif market_cap < 100_000_000:
        return 4
    elif market_cap < 250_000_000:
        return 3
    elif market_cap < 1_000_000_000:
        return 2
    elif market_cap > 10_000_000_000:
        return 1
    return 5

def check_and_fetch_info(ticker):
    market_cap = get_market_cap(ticker)
    days_to_check = get_days_to_check(market_cap)

    # Construct the file path 
    if isinstance(ticker, str):
        first_letter = ticker[0].upper()
    else:
        # Handle the case where ticker is not a string
        print(f"Unexpected type for ticker: {type(ticker)}")
        return

    directory = f"Datasets/Ticker/{first_letter}"
    filename = f"{directory}/{ticker}.info"

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # Check if the file exists and its modification time
    if os.path.exists(filename):
        # Get the current time and the file's last modification time
        current_time = time.time()
        file_mod_time = os.path.getmtime(filename)
        
        # Calculate the age of the file
        file_age_days = (current_time - file_mod_time) / (60 * 60 * 24)
        
        # Check if the file is older than the days_to_check
        if file_age_days <= days_to_check:
            print(f"Info file for {ticker} is up-to-date (less than {days_to_check} days old). Skipping...")
            return

    # Fetch the info since the file doesn't exist or is older than the days_to_check
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
        time.sleep(1)
    except Exception as e:
        logging.error(f"Error fetching info for {ticker}: {e}")

def main():
    # Example: Load a list of tickers from a CSV file
    df = pd.read_csv('Datasets/us_stocks.csv')

    print(df.columns)

    tickers = df['symbol'].tolist()

    for ticker in tickers:
        check_and_fetch_info(ticker)

if __name__ == "__main__":
    main()
