#fetch_us_stocks.py

import finnhub
import pandas as pd
import csv

# Your Finnhub API key
API_KEY = "cqa6h0hr01qkfes2n7ogcqa6h0hr01qkfes2n7p0"

def fetch_us_stocks():
    # Initialize the Finnhub client
    finnhub_client = finnhub.Client(api_key=API_KEY)

    # Fetch all stock symbols for the US market
    us_stocks = finnhub_client.stock_symbols('US')

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(us_stocks)

    # Write the DataFrame to a CSV file
    output_file = 'us_stocks.csv'
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"US stocks data saved to {output_file}")

if __name__ == "__main__":
    fetch_us_stocks()
