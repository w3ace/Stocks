import argparse
import pandas as pd
from datetime import timedelta
from fetch_stock import fetch_stock


def fetch_intraday(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "1m",
) -> pd.DataFrame:
    """Fetch intraday data for ``ticker`` using :func:`fetch_stock`."""

    data, _ = fetch_stock(ticker, start_date=start, end_date=end, interval=interval)
    if data is None or data.empty:
        return pd.DataFrame()

    if "Datetime" in data.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(data["Datetime"], errors="coerce"))
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(data.index, errors="coerce"))

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    data.index = idx
    if "Datetime" in data.columns:
        data = data.drop(columns=["Datetime"])

    data.sort_index(inplace=True)
    data = data.loc[(data.index >= start) & (data.index <= end + timedelta(days=1))]
    return data


def calculate_morning_range(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with daily high/low from 9:30 to 10:00 US/Eastern."""
    df = df.tz_convert("US/Eastern")
    morning = df.between_time("09:30", "10:00")
    if morning.empty:
        return pd.DataFrame()
    grouped = morning.groupby(morning.index.date)[["High", "Low"]].agg({"High": "max", "Low": "min"})
    grouped.index = pd.to_datetime(grouped.index)
    grouped.sort_index(inplace=True)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate high/low from 9:30 to 10:00am")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    start = pd.to_datetime(args.start).tz_localize("UTC")
    end = pd.to_datetime(args.end).tz_localize("UTC")

    for ticker in args.tickers:
        data = fetch_intraday(ticker, start, end, interval="1m")
        if data.empty:
            print(f"No 1m data for {ticker}. Trying 5m interval...")
            data = fetch_intraday(ticker, start, end, interval="5m")
        if data.empty:
            print(f"Unable to fetch intraday data for {ticker} in given range.")
            continue
        result = calculate_morning_range(data)
        if result.empty:
            print(f"No trading data between 9:30 and 10:00 for {ticker}.")
            continue
        print(f"\nMorning range for {ticker}:")
        print(result)


if __name__ == "__main__":
    main()