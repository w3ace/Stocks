import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

def stochastic_oscillator(df, k_period, d_period):
    """
    Calculate the Fast Stochastic Oscillator.
    :param df: DataFrame with columns 'High', 'Low', 'Close'
    :param k_period: Period for %K calculation
    :param d_period: Smoothing period for %D
    :return: DataFrame with %K and %D values
    """
    df['L'] = df['Low'].rolling(window=k_period).min()
    df['H'] = df['High'].rolling(window=k_period).max()
    df['%K'] = 100 * ((df['Close'] - df['L']) / (df['H'] - df['L']))
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    return df[['%K', '%D']]

def full_stochastic_oscillator(df, k_period, k_slow_period, d_period):
    """
    Calculate the Full Stochastic Oscillator.
    :param df: DataFrame with columns 'High', 'Low', 'Close'
    :param k_period: Period for %K calculation
    :param k_slow_period: Smoothing period for %K (slow)
    :param d_period: Smoothing period for %D
    :return: DataFrame with %K and %D values
    """
    df['L'] = df['Low'].rolling(window=k_period).min()
    df['H'] = df['High'].rolling(window=k_period).max()
    df['%K'] = 100 * ((df['Close'] - df['L']) / (df['H'] - df['L']))
    df['%K_slow'] = df['%K'].rolling(window=k_slow_period).mean()
    df['%D'] = df['%K_slow'].rolling(window=d_period).mean()
    return df[['%K_slow', '%D', '%K']]

def fetch_and_calculate_stochastic(ticker, start_date, end_date):
    """
    Fetch ticker data from yfinance and calculate stochastic oscillator.
    :param ticker: Ticker symbol
    :param start_date: Start date for data
    :param end_date: End date for data
    """
    # Download 5-min interval data
    df = yf.download(tickers=ticker, start=start_date, end=end_date, interval="5m", prepost=True)
    df = df.dropna()

    # Calculate multiple stochastic values
    stochastic_values = {
        '(9,3)': stochastic_oscillator(df.copy(), 9, 3),
        '(14,3)': stochastic_oscillator(df.copy(), 14, 3),
        '(40,4)': stochastic_oscillator(df.copy(), 40, 4),
        '(60,10)': full_stochastic_oscillator(df.copy(), 60, 3, 10),
    }

    # Combine all results
    results = pd.DataFrame(index=df.index)
    for label, stochastic_df in stochastic_values.items():
        results[f'%K {label}'] = stochastic_df['%K_slow'] if 'slow' in stochastic_df.columns else stochastic_df['%K']
        results[f'%D {label}'] = stochastic_df['%D']

    return df, results

def plot_candles_and_stochastics(df, stochastics, ticker):
    """
    Plot price candles, volume, and stochastic oscillator values for the last five days.
    Add thin vertical lines between days for clarity.
    Generate graphs only if %D (60,10), %D (40,4), and %D (14,3) are below 20.
    :param df: DataFrame with Open, High, Low, Close, Volume data
    :param stochastics: DataFrame with %K values for different stochastic periods
    :param ticker: Ticker symbol for the title
    """
    # Filter for the last five trading days
    #df = df[-(78 * 5):]
    #stochastics = stochastics[-(78 * 5):]

    # Check if the last %D (60,10), %D (40,4), and %D (14,3) values are below 20
    if not (stochastics['%D (60,10)'].iloc[-1] < 20 and \
            stochastics['%D (40,4)'].iloc[-1] < 20 and \
            stochastics['%D (14,3)'].iloc[-1] < 20):
        print(f"Skipping {ticker}: Last %D (60,10), %D (40,4), or %D (14,3) value is not below 20.")
        return False

    plt.style.use('dark_background')
    plt.figure(figsize=(14, 24))

    # Combine x-axis indices for continuity across five days
    df['Index'] = range(len(df))
    stochastics['Index'] = range(len(stochastics))

    # Find day boundaries
    day_boundaries = df.index.normalize().unique()
    boundary_indices = [df[df.index.normalize() == day].iloc[0]['Index'] for day in day_boundaries]

    # Plot price candles as bar candles (larger subplot)
    plt.subplot(6, 1, (1, 2))
    for idx, row in df.iterrows():
        color = 'green' if row['Close'] >= row['Open'] else 'red'
        plt.bar(row['Index'], row['Close'] - row['Open'], bottom=row['Open'], color=color, width=0.4)
        plt.vlines(row['Index'], row['Low'], row['High'], color=color)
    for boundary in boundary_indices:
        plt.axvline(boundary, color='white', linestyle='--', linewidth=0.5)
    plt.title(f'\n\nPrice Candles for {ticker} (Last 5 Days)', color='lightgrey')
    plt.ylabel('Price', color='lightgrey')
    plt.tick_params(colors='lightgrey')

    # Plot volume
    plt.subplot(6, 1, 3)
    plt.bar(df['Index'], df['Volume'], color='lightblue', alpha=0.6)
    for boundary in boundary_indices:
        plt.axvline(boundary, color='white', linestyle='--', linewidth=0.5)
    plt.title('Volume', color='lightgrey')
    plt.ylabel('Volume', color='lightgrey')
    plt.tick_params(colors='lightgrey')

    # Plot (9,3) and (14,3) stochastics
    plt.subplot(6, 1, 4)
    plt.plot(stochastics['Index'], stochastics['%D (9,3)'], color='red', label='%D (9,3)')
    plt.plot(stochastics['Index'], stochastics['%D (14,3)'], color='green', label='%D (14,3)')
    for boundary in boundary_indices:
        plt.axvline(boundary, color='white', linestyle='--', linewidth=0.5)
    plt.axhline(20, color='gray', linestyle='--', alpha=0.7)
    plt.ylabel('Stochastic %D', color='lightgrey')
    plt.legend()
    plt.tick_params(colors='lightgrey')

    # Plot (40,4) and (60,10) stochastics
    plt.subplot(6, 1, 5)
    plt.plot(stochastics['Index'], stochastics['%D (40,4)'], color='purple', label='%D (40,4)')
    plt.plot(stochastics['Index'], stochastics['%D (60,10)'], color='gold', label='%D (60,10)')
    for boundary in boundary_indices:
        plt.axvline(boundary, color='white', linestyle='--', linewidth=0.5)
    plt.axhline(20, color='gray', linestyle='--', alpha=0.7)
    plt.ylabel('Stochastic %D', color='lightgrey')
    plt.xlabel('Data Points', color='lightgrey')
    plt.legend()
    plt.tick_params(colors='lightgrey')

    plt.tight_layout()
    plt.show()
    return True

# Example usage
input_tickers = input("Enter ticker symbols (separated by spaces or commas): ").upper()
tickers = [ticker.strip() for ticker in input_tickers.replace(',', ' ').split()]
start_date = pd.Timestamp.now() - pd.Timedelta(days=8)
end_date = pd.Timestamp.now()

successfully_fetched = []
matched_criteria = []

for ticker in tickers:
    print(f"\nProcessing {ticker}...")
    try:
        price_data, stochastic_data = fetch_and_calculate_stochastic(ticker, start_date, end_date)
        successfully_fetched.append(ticker)
    except Exception as fetch_error:
        print(f"Error fetching or calculating stochastic data for {ticker}:", fetch_error)
        continue

    try:
        if plot_candles_and_stochastics(price_data, stochastic_data, ticker):
            matched_criteria.append(ticker)
    except Exception as plot_error:
        print(f"Error plotting data for {ticker}:", plot_error)

print("\nSuccessfully fetched tickers:", ", ".join(successfully_fetched))
print("Matched criteria tickers:", ", ".join(matched_criteria))
