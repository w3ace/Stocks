import yfinance as yf
import pandas as pd
import os
import time
import logging

# Set up logging
logging.basicConfig(filename='fetch_stock_info.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def fetch_and_save_stock_info(ticker):
    try:

    first_letter = ticker[0].upper()
    directory = f"Datasets/Ticker/{first_letter}"
    filename = f"{directory}/{ticker}.info"

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filename):
            # Get the current time and the file's last modification time
            current_time = time.time()
            file_mod_time = os.path.getmtime(filename)
            
            # Calculate the age of the file
            file_age_days = (current_time - file_mod_time) / (60 * 60 * 24)
            
            # Check if the file is older than 5 days
            if file_age_days <= 5:
                print(f"Info file for {ticker} is up-to-date (less than 5 days old). Skipping...")
                return       # Fetch company profile and financial information from yfinance
 
        stock = yf.Ticker(ticker)
        info = stock.info

        # Save the information to a file
        with open(filename, 'w') as file:
            for key, value in info.items():
                file.write(f"{key}: {value}\n")
        print(f"Info file for {ticker} saved.")
        time.sleep(2)  # Wait for 10 seconds before the next call
    
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        print(f"Error fetching data for {ticker}: {e}")

def main():
    # Load the list of stocks from the CSV file
    csv_file = 'Datasets/us_stocks.csv'
    stocks_df = pd.read_csv(csv_file)

    # Ensure that the ticker symbols are treated as strings
    stocks_df['symbol'] = stocks_df['symbol'].astype(str)

    # Iterate over each stock and fetch its information
    for ticker in stocks_df['symbol']:
        fetch_and_save_stock_info(ticker)

if __name__ == "__main__":
    main()
