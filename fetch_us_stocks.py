#fetch_us_stocks.py

import finnhub
import pandas as pd
import csv
from pathlib import Path

# Read the Finnhub API key from API_KEY.txt located in the project root.
# The file should contain only the API key string and must not be tracked by
# version control.
API_KEY_PATH = Path(__file__).resolve().parent / "API_KEY.txt"
try:
    API_KEY = API_KEY_PATH.read_text().strip()
except FileNotFoundError:
    raise FileNotFoundError(
        f"API key file not found: {API_KEY_PATH}. Create this file with your Finnhub API key."
    )

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
