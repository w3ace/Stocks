import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from fetch_stock import fetch_stock

logging.basicConfig(level=logging.INFO)


def fetch_close_prices(symbol, start_date=None, end_date=None, period="1mo"):
    """Return daily closing price series for ``symbol``."""
    if start_date is not None and end_date is not None:
        _, daily = fetch_stock(symbol, start_date=start_date, end_date=end_date)
    else:
        _, daily = fetch_stock(symbol, period=period)

    if daily is None or daily.empty:
        logging.warning(f"No data for {symbol}")
        return pd.Series(dtype=float)

    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily.sort_values("Date", inplace=True)
    daily.set_index("Date", inplace=True)
    return daily["Close"]


def main():
    parser = argparse.ArgumentParser(
        description="Plot cumulative percent change for an equal-weighted portfolio compared to SPY, QQQ, and IWM."
    )
    parser.add_argument("symbols", nargs="+", help="Portfolio ticker symbols")
    parser.add_argument("--start", type=str, help="Start date mm-dd-yyyy")
    parser.add_argument("--end", type=str, help="End date mm-dd-yyyy")
    parser.add_argument("--period", type=str, help="Period string (e.g., 6mo, 1y)")
    args = parser.parse_args()

    if args.start and args.end:
        start_date = pd.to_datetime(args.start, format="%m-%d-%Y")
        end_date = pd.to_datetime(args.end, format="%m-%d-%Y")
    elif args.period:
        start_date = end_date = None
    else:
        print("Please provide either a date range or a period.")
        return

    closes_df = pd.DataFrame()
    for ticker in args.symbols:
        series = fetch_close_prices(ticker, start_date, end_date, args.period)
        if not series.empty:
            closes_df[ticker] = series

    if closes_df.empty:
        print("No data downloaded for the given tickers.")
        return

    norm_closes = closes_df.divide(closes_df.iloc[0])
    portfolio_value = norm_closes.mean(axis=1)
    portfolio_change = (portfolio_value - 1) * 100

    spy_close = fetch_close_prices("SPY", start_date, end_date, args.period)
    qqq_close = fetch_close_prices("QQQ", start_date, end_date, args.period)
    iwm_close = fetch_close_prices("IWM", start_date, end_date, args.period)

    spy_change = (spy_close / spy_close.iloc[0] - 1) * 100 if not spy_close.empty else pd.Series(dtype=float)
    qqq_change = (qqq_close / qqq_close.iloc[0] - 1) * 100 if not qqq_close.empty else pd.Series(dtype=float)
    iwm_change = (iwm_close / iwm_close.iloc[0] - 1) * 100 if not iwm_close.empty else pd.Series(dtype=float)

    comparison = pd.concat(
        [
            portfolio_change.rename("Portfolio"),
            spy_change.rename("SPY"),
            qqq_change.rename("QQQ"),
            iwm_change.rename("IWM"),
        ],
        axis=1,
    )
    comparison.sort_index(inplace=True)

    comparison.plot(figsize=(10, 6))
    plt.title("Cumulative Percentage Change")
    plt.xlabel("Date")
    plt.ylabel("Cumulative % Change")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
