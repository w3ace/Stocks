import pandas as pd

def analyze_stock(hourly_data, daily_data):
    # Calculate the difference between the open and close prices for each day
    daily_data['Open_Close_Diff'] = daily_data['Close'] - daily_data['Open']
    
    # Calculate the difference between the close price of one day and the open price of the next day
    daily_data['Close_NextOpen_Diff'] = daily_data['Open'].shift(-1) - daily_data['Close']
    
    
    # Calculate cumulative sums for the differences starting from the initial open and close prices
    daily_data['Cumulative_Open_Close_Diff'] = daily_data['Open_Close_Diff'].cumsum() + daily_data['Open'].iloc[0]
    daily_data['Cumulative_Close_NextOpen_Diff'] = daily_data['Close_NextOpen_Diff'].cumsum() + daily_data['Close'].iloc[0]

    return hourly_data, daily_data
