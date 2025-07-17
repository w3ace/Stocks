import yfinance as yf
import pandas as pd

from stock_functions import round_numeric_cols

# Fetch data for a specific stock symbol
symbol = 'AAPL'
stock = yf.Ticker(symbol)

# Convert the info dictionary to a DataFrame
info_df = pd.DataFrame.from_dict(stock.info, orient='index', columns=['Value'])
info_df = round_numeric_cols(info_df)

# Print the DataFrame
print(info_df)

# Alternatively, to write it to a CSV file:
info_df.to_csv(f'{symbol}_info.csv')
