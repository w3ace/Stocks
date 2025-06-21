import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_friday_monday_data(ticker, start_date, num_weeks):
    # Convert start date to datetime and calculate end date
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = start_date + timedelta(weeks=num_weeks)

    # Download data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    stock_data = stock_data[['Close']]

    # Filter to only include Fridays and Mondays
    friday_monday_data = stock_data[stock_data.index.dayofweek.isin([4, 0])]

    # Initialize variables
    results = []
    cumulative_gain_loss = 0  # Cumulative gain/loss for Friday to Monday
    cumulative_friday_to_friday = 0  # Cumulative Friday-to-Friday change
    fridays = friday_monday_data[friday_monday_data.index.dayofweek == 4]
    
    # Loop over Fridays to calculate cumulative gains/losses
    prev_friday_close = None  # Track the previous Friday close for percentage change calculation

    for friday_date, friday_close in fridays['Close'].items():
        monday_date = friday_date + timedelta(days=3)
        
        # Check if Monday's close price exists
        if monday_date in friday_monday_data.index:
            monday_close = friday_monday_data.loc[monday_date, 'Close']
            gain_loss = (monday_close - friday_close) / friday_close
            cumulative_gain_loss += gain_loss  # Update cumulative gain/loss

            # Calculate and accumulate Friday-to-Friday change
            if prev_friday_close is not None:
                friday_to_friday_change = (friday_close - prev_friday_close) / prev_friday_close
                cumulative_friday_to_friday += friday_to_friday_change
            else:
                friday_to_friday_change = 0  # No previous Friday for the first iteration

            prev_friday_close = friday_close  # Update previous Friday close for next iteration
            
            results.append({
                'Friday': friday_date,
                'Monday': monday_date,
                'Friday Close': friday_close,
                'Monday Close': monday_close,
                'Gain/Loss %': gain_loss * 100,
                'Cumulative Gain/Loss %': cumulative_gain_loss * 100,  # Cumulative %
                'Friday-to-Friday Change %': friday_to_friday_change * 100,  # Weekly %
                'Cumulative Friday-to-Friday %': cumulative_friday_to_friday * 100  # Cumulative Friday-to-Friday %
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def plot_cumulative_gain_loss_with_friday_to_friday(df, ticker):
    plt.figure(figsize=(12, 6))

    # Plot cumulative gain/loss from Friday to Monday close
    plt.plot(df['Monday'], df['Cumulative Gain/Loss %'], marker='o', linestyle='-', label='Cumulative Gain/Loss % (Friday to Monday Close)')

    # Plot cumulative Friday-to-Friday change
    plt.plot(df['Monday'], df['Cumulative Friday-to-Friday %'], marker='x', linestyle='-', label="Cumulative Friday-to-Friday Change %", color='orange')

    plt.title(f'Cumulative Gains/Losses and Friday-to-Friday Changes for {ticker}')
    plt.xlabel('Monday')
    plt.ylabel('Cumulative Percentage Change')
    plt.legend()
    plt.grid(True)
    plt.show()

# Inputs
ticker = 'spy'  # Replace with desired ticker
start_date = '2021-01-01'  # Replace with start date in 'YYYY-MM-DD' format
num_weeks = 190  # Replace with desired number of weeks

# Get data and plot
df = get_friday_monday_data(ticker, start_date, num_weeks)
plot_cumulative_gain_loss_with_friday_to_friday(df, ticker)
