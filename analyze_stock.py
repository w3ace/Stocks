import pandas as pd
import numpy as np

def analyze_stock(hourly_data, daily_data):
    # Ensure we have a copy of the DataFrame
    daily_data = daily_data.copy()

    # Calculate the difference between the open and close prices for each day
    daily_data['Open_Close_Diff'] = daily_data['Close'] - daily_data['Open']

    # Calculate the difference between the close price of one day and the open price of the next day
    daily_data['Close_NextOpen_Diff'] = daily_data['Open'].shift(-1) - daily_data['Close']

    # Calculate gap between the previous day's close and today's open
    daily_data['PrevClose_Open_Gap'] = daily_data['Open'] - daily_data['Close'].shift(1)
    daily_data['PrevClose_Open_Gap_Pct'] = (
        daily_data['PrevClose_Open_Gap'] / daily_data['Close'].shift(1)
    ) * 100
    
    # Remove the last row which will have NaN value for Close_NextOpen_Diff
    daily_data = daily_data[:-1]
    
    # Calculate cumulative sums for the differences starting from the initial open and close prices
    daily_data['Cumulative_Open_Close_Diff'] = daily_data['Open_Close_Diff'].cumsum() + daily_data['Open'].iloc[0]
    daily_data['Cumulative_Close_NextOpen_Diff'] = daily_data['Close_NextOpen_Diff'].cumsum() + daily_data['Open'].iloc[0]
    
    # Calculate MACD
    short_ema = daily_data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = daily_data['Close'].ewm(span=26, adjust=False).mean()
    daily_data['MACD'] = short_ema - long_ema
    daily_data['Signal_Line'] = daily_data['MACD'].ewm(span=9, adjust=False).mean()

    # Classify first hour trading data
    hourly_data['Date'] = hourly_data['Datetime'].dt.date
    first_hour_data = hourly_data.groupby('Date').first().reset_index()
    first_hour_high = hourly_data.groupby('Date')['High'].max().reset_index(name='First_Hour_High')
    first_hour_low = hourly_data.groupby('Date')['Low'].min().reset_index(name='First_Hour_Low')
    
    first_hour_data = first_hour_data.merge(first_hour_high, on='Date')
    first_hour_data = first_hour_data.merge(first_hour_low, on='Date')

    # Rename columns in first_hour_data to avoid conflicts
    first_hour_data = first_hour_data.rename(columns={
        'Open': 'First_Hour_Open',
        'Close': 'First_Hour_Close'
    })

    # Ensure 'Date' columns are of the same type
    daily_data['Date'] = pd.to_datetime(daily_data['Date']).dt.date
    first_hour_data['Date'] = pd.to_datetime(first_hour_data['Date']).dt.date

    daily_data = daily_data.merge(first_hour_data[['Date', 'First_Hour_Open', 'First_Hour_Close', 'First_Hour_High', 'First_Hour_Low']], left_on='Date', right_on='Date', how='left')

    daily_data['First_Hour_Range'] = daily_data['First_Hour_High'] - daily_data['First_Hour_Low']
    daily_data['First_Hour_Close_Change'] = (daily_data['First_Hour_Close'] - daily_data['First_Hour_Open']) / daily_data['First_Hour_Open'] * 100

    conditions = [
        daily_data['First_Hour_Range'] > 0.05 * daily_data['Open'],
        daily_data['First_Hour_Close_Change'] > 2.5,
        daily_data['First_Hour_Close_Change'] < -2.5,
        daily_data['First_Hour_Range'] > 0.025 * daily_data['Open']
    ]
    choices = [
        'First Hour Range > 5%',
        'First Hour Close > 2.5% higher than Open',
        'First Hour Close > 2.5% lower than Open',
        'First Hour Range > 2.5%'
    ]
    daily_data['First_Hour_Classification'] = np.select(conditions, choices, default='Other')

    return hourly_data, daily_data

