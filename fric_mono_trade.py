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
    stock_data = stock_data[['Close','Open']]

    # Filter to only include Fridays and Mondays
    friday_monday_data = stock_data[stock_data.index.dayofweek.isin([4, 0])]

    print (friday_monday_data)

    # Initialize variables
    results = []
    cumulative_gain_loss = 0  # Cumulative gain/loss for Friday to Monday
    cumulative_friday_to_friday = 0  # Cumulative Friday-to-Friday change
    cumulative_monday_to_friday = 0
    fridays = friday_monday_data[friday_monday_data.index.dayofweek == 4]
    
    # Loop over Fridays to calculate cumulative gains/losses
    prev_friday_close = None  # Track the previous Friday close for percentage change calculation
    prev_monday_open = None 

    for friday_date, friday_close in fridays['Close'].items():
        monday_date = friday_date + timedelta(days=3)
        
        # Check if Monday's open price exists
        if monday_date in friday_monday_data.index:
            monday_open = friday_monday_data.loc[monday_date, 'Open']
            gain_loss = (monday_open - friday_close) / friday_close
            cumulative_gain_loss += gain_loss  # Update cumulative gain/loss

            # Calculate and accumulate Friday-to-Friday change
            if prev_friday_close is not None:
                friday_to_friday_change = (friday_close - prev_friday_close) / prev_friday_close
                cumulative_friday_to_friday += friday_to_friday_change
            else:
                friday_to_friday_change = 0  # No previous Friday for the first iteration

            if prev_monday_open is not None:
                # Calculate Monday open to Friday close gain/loss
                monday_to_friday_gain = (friday_close - prev_monday_open) / prev_monday_open
                cumulative_monday_to_friday += monday_to_friday_gain  # Update cumulative Monday to Friday
            else:
                monday_to_friday_gain = 0  # No previous Friday for the first iteration

            prev_friday_close = friday_close  # Update previous Friday close for next iteration
            prev_monday_open = monday_open
               
    
            results.append({
                'Friday': friday_date,
                'Monday': monday_date,
                'Friday Close': friday_close,
                'Monday Open': monday_open,
                'Gain/Loss %': gain_loss * 100,
                'Cumulative Gain/Loss %': cumulative_gain_loss * 100,  # Cumulative %
                'Friday-to-Friday Change %': friday_to_friday_change * 100,  # Weekly %
                'Cumulative Friday-to-Friday %': cumulative_friday_to_friday * 100,  # Cumulative Friday-to-Friday %
                'Cumulative Monday-to-Friday %': cumulative_monday_to_friday * 100  # Cumulative Monday-to-Friday %
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def plot_cumulative_gain_loss(df, ticker):
    plt.figure(figsize=(12, 6))

    # Plot cumulative gain/loss from Friday to Monday
    plt.plot(df['Monday'], df['Cumulative Gain/Loss %'], marker='o', linestyle='-', label='Cumulative Gain/Loss %')

    # Plot cumulative Friday-to-Friday change
    plt.plot(df['Monday'], df['Cumulative Friday-to-Friday %'], marker='x', linestyle='-', label="Cumulative Friday-to-Friday Change %", color='orange')

    # Plot cumulative gain/loss from Monday open to Friday close
    plt.plot(df['Monday'], df['Cumulative Monday-to-Friday %'], marker='s', linestyle='-', label="Cumulative Monday-to-Friday Gain %", color='green')

    plt.title(f'Cumulative Gains/Losses and Friday-to-Friday Changes for {ticker}')
    plt.xlabel('Monday')
    plt.ylabel('Cumulative Percentage Change')
    plt.legend()
    plt.grid(True)
    plt.show()

# Inputs
tickers = 'xbi,ibm,nvda,msft'  # Replace with desired ticker
start_date = '2024-01-01'  # Replace with start date in 'YYYY-MM-DD' format
num_weeks = 52  # Replace with desired number of weeks

# Get data and plot
#df = get_friday_monday_data(ticker, start_date, num_weeks)
#plot_cumulative_gain_loss_with_friday_to_friday(df, ticker)

# Process each ticker and store results in a dictionary
for ticker in tickers.split(','):
    ticker = ticker.strip()
    df = get_friday_monday_data(ticker, start_date, num_weeks)
    print (df)
    plot_cumulative_gain_loss(df,ticker) # Plot cumulative gains/losses for each ticker