import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import argparse

def fetch_stock_data(symbol, period):
    # Fetch the historical data for the specified period
    stock_data = yf.download(symbol, period=period)
    
    # Replace NaN values with 0
    stock_data.fillna(0, inplace=True)
    
    # Calculate the opening-closing gap for each day
    stock_data['Opening_Closing_Gap'] = stock_data['Close'] - stock_data['Open']

    # Calculate the previous day's close to current day's open gap
    stock_data['Previous_Close_to_Open_Gap'] = stock_data['Open'] - stock_data['Close'].shift(1)

    # Calculate cumulative sums for each gap column
    stock_data['Cumulative_Opening_Closing_Gap'] = stock_data['Opening_Closing_Gap'].cumsum()
    stock_data['Cumulative_Previous_Close_to_Open_Gap'] = stock_data['Previous_Close_to_Open_Gap'].cumsum()

    # Calculate cumulative sum of the two gaps
    stock_data['Cumulative_Total_Gap'] = stock_data['Cumulative_Opening_Closing_Gap'] + stock_data['Cumulative_Previous_Close_to_Open_Gap']

    return stock_data

def plot_stock_data(stock_data, symbol):
    # Plotly visualization
    fig = go.Figure()

    # Add trace for Cumulative Opening-Closing Gap
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Cumulative_Opening_Closing_Gap'],
                             mode='lines',
                             name='Cumulative Opening-Closing Gap'))

    # Add trace for Cumulative Previous Close-Open Gap
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Cumulative_Previous_Close_to_Open_Gap'],
                             mode='lines',
                             name='Cumulative Previous Close-Open Gap'))

    # Add trace for Cumulative Total Gap
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Cumulative_Total_Gap'],
                             mode='lines',
                             name='Cumulative Total Gap'))

    # Update layout
    fig.update_layout(title=f'{symbol} Cumulative Gap Totals',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Gap Amount',
                      legend=dict(x=0, y=1, traceorder='normal'))

    # Show the plot
    fig.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fetch and plot cumulative gap totals for a stock symbol.')
    parser.add_argument('symbol', type=str, help='Stock symbol (e.g., SPY)')
    parser.add_argument('--period', type=str, default='1mo', help='Time period for data (e.g., 1mo, 1d, 1y, etc.)')
    args = parser.parse_args()

    # Fetch stock data
    stock_data = fetch_stock_data(args.symbol, args.period)

    # Plot stock data
    plot_stock_data(stock_data, args.symbol)
