import numpy as np
import pandas as pd
import ta

def analyze_stock(stock_data, start_date=None, end_date=None):


    stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'])
    stock_data['Date'] = stock_data['Datetime'].dt.date
    stock_data['Time'] = stock_data['Datetime'].dt.time

 
    initial_open = stock_data['Open'].iloc[0]  # Opening price for the first day

    # Calculate cumulative gaps based on available columns
    daily_stock_data['Open to Close Cumulative'] = (daily_stock_data['Close'] - daily_stock_data['Open']).cumsum() + initial_open

    # Handle cases where daily_stock_data['Open'].shift(-1) is NaN
    shifted_open = daily_stock_data['Open'].shift(-1)
    close_to_open = np.where(shifted_open.isna(), daily_stock_data['Close'], shifted_open)
    
    # Calculate cumulative difference using the modified 'close_to_open' array
    daily_stock_data['Close to Open Cumulative'] = (close_to_open - daily_stock_data['Close']).cumsum() + initial_open

    # Calculate percentage change from first day opening for cumulative gaps
    daily_stock_data['Open to Close % Change'] = (daily_stock_data['Open to Close Cumulative'] - initial_open) / initial_open * 100
    daily_stock_data['Close to Open % Change'] = (stock_data['Close to Open Cumulative'] - initial_open) / initial_open * 100

    # Calculate Moving Average Convergence Divergence (MACD) within the specified date range
    if start_date and end_date:
        stock_data_within_range = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
        macd = ta.trend.macd(stock_data_within_range['Close'])
        macd_signal = ta.trend.macd_signal(stock_data_within_range['Close'])
        stock_data.loc[stock_data_within_range.index, 'MACD'] = macd
        stock_data.loc[stock_data_within_range.index, 'MACD_Signal'] = macd_signal
    else:
        stock_data['MACD'] = ta.trend.macd(stock_data['Close'])
        stock_data['MACD_Signal'] = ta.trend.macd_signal(stock_data['Close'])

    # Print the data table to the terminal with updated column headings
    print(f"\nCumulative Gap Totals:\n")
    print(stock_data[['Open to Close Cumulative', 'Open to Close % Change', 'Close to Open Cumulative', 'Close to Open % Change', 'Open']])

    return daily_stock_data,stock_data
