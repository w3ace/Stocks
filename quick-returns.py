import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_returns(ticker, auto_adjust=False):
    """
    Calculate stock returns for different time periods.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        auto_adjust (bool): Whether to adjust for splits/dividends

    Returns:
        dict: Returns for different periods
    """
    # Initialize return dictionary
    returns = {}

    # Validate ticker
    if len(ticker) < 3:
        print(f"Error: Invalid ticker '{ticker}'")
        return None

    # Fetch historical stock data
    try:
        data = yf.download(ticker, period="5y", auto_adjust=auto_adjust)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

    if data.empty:
        print(f"No data available for {ticker}")
        return None

    # Close price series
    close = data['Close']

    # Define periods in days (approximately)
    periods = {
        "1 wk": 5,
        "1 month": 21,
        "1 year": 252,
        "3 year": 780
    }

    # Calculate returns for each period
    for label, days in periods.items():
        # Ensure minimum days in data before calculating
        if len(close) < days:
            returns[label] = None
            continue

        # Get start and end dates
        start_idx = days - 1
        end_idx = len(close) - 1

        start_price = close.iloc[start_idx]
        end_price = close.iloc[end_idx]

        # Calculate percentage return
        if start_price == 0:
            return_pct = 0 if end_price == 0 else 100
        else:
            return_pct = ((end_price - start_price) / start_price) * 100

        returns[label] = return_pct

    return returns


def get_stock_info(ticker):
    """
    Get additional stock information (name, currency, etc.)
    """
    try:
        info = yf.Ticker(ticker)
        return info.info
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return {}


def display_results(ticker, returns, stock_info):
    """
    Display calculated results in a formatted way.
    """
    print(f"\n{'='*60}")
    print(f"STOCK: {ticker}")
    print(f"{'='*60}")

    # Display stock info
    print(f"Company Name: {stock_info.get('longName', 'N/A')}")
    print(f"Industry: {stock_info.get('industry', 'N/A')}")
    print(f"CEO: {stock_info.get('managingDirector', 'N/A')}")
    print(f"Currency: {stock_info.get('currency', 'N/A')}")
    print(f"{'='*60}")

    # Display returns
    print(f"\nSTOCK RETURNS:")
    print(f"{'='*60}")

    for label, pct in returns.items():
        if pct is not None:
            pct_str = f"{pct:+.2f}%"
        else:
            pct_str = "N/A"

        print(f"{label:8} | {pct_str:8}")

    # Summary
    total = returns.get('3 year', 0)
    print(f"\n{'='*60}")
    print(f"TOTAL 3-YEAR RETURN: {total:+.2f}%")
    print(f"{'='*60}")


def main():
    """
    Main function to run the script.
    """
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()

    # Fetch stock data
    returns = calculate_returns(ticker, auto_adjust=False)

    if returns is None:
        print("Error: Could not calculate returns")
        return

    # Get additional stock info
    stock_info = get_stock_info(ticker)

    # Display results
    display_results(ticker, returns, stock_info)

    # Ask user if they want to run again
    print("\n" + "=" * 60)
    run_again = input("Run again? (y/n): ").strip().lower()
    if run_again == 'y':
        main()


if __name__ == "__quick-returns.py":
    main()
