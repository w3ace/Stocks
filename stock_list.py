#stock_list.py

import finnhub

finnhub_client = finnhub.Client(api_key="cqa6h0hr01qkfes2n7ogcqa6h0hr01qkfes2n7p0")


# Fetch all stock symbols for the US market, which includes NYSE
all_us_stocks = finnhub_client.stock_symbols('US')

# Filter out only NYSE stocks
nyse_stocks = [stock for stock in all_us_stocks if stock['exchange'] == 'NYSE']

# Print the list of NYSE stocks
for stock in nyse_stocks:
    print(f"Symbol: {stock['symbol']}, Description: {stock['description']}, Exchange: {stock['exchange']}")

