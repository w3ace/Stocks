#stock_list.py

import finnhub
from pathlib import Path

# Read the Finnhub API key from API_KEY.txt located in the project root.
API_KEY_PATH = Path(__file__).resolve().parent / "API_KEY.txt"
try:
    API_KEY = API_KEY_PATH.read_text().strip()
except FileNotFoundError:
    raise FileNotFoundError(
        f"API key file not found: {API_KEY_PATH}. Create this file with your Finnhub API key."
    )

finnhub_client = finnhub.Client(api_key=API_KEY)


# Fetch all stock symbols for the US market, which includes NYSE
all_us_stocks = finnhub_client.stock_symbols('US')

# Filter out only NYSE stocks
nyse_stocks = [stock for stock in all_us_stocks if stock['exchange'] == 'NYSE']

# Print the list of NYSE stocks
for stock in nyse_stocks:
    print(f"Symbol: {stock['symbol']}, Description: {stock['description']}, Exchange: {stock['exchange']}")

