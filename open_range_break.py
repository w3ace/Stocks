import argparse
from datetime import timedelta

import pandas as pd

from fetch_stock import fetch_stock
from stock_functions import choose_yfinance_interval


def fetch_intraday(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str = "5m") -> pd.DataFrame:
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


def analyze_open_range(df: pd.DataFrame) -> tuple[int, int, int]:
    """Analyze opening range breaks for each trading day.

    Returns tuple of (total_days, broke_low_first, broke_low_then_high).
    """
    if df.empty:
        return 0, 0, 0

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)

    total_days = 0
    broke_low_first = 0
    broke_low_then_high = 0

    for date, day_df in grouped:
        morning = day_df.between_time("09:30", "10:00")
        if morning.empty:
            continue
        or_high = morning["High"].max()
        or_low = morning["Low"].min()
        after_open = day_df[day_df.index > morning.index[-1]]
        if after_open.empty:
            total_days += 1
            continue

        high_cross_time = None
        low_cross_time = None
        for idx, row in after_open.iterrows():
            if low_cross_time is None and row["Low"] <= or_low:
                low_cross_time = idx
            if high_cross_time is None and row["High"] >= or_high:
                high_cross_time = idx
            if low_cross_time is not None and high_cross_time is not None:
                break

        total_days += 1
        if low_cross_time is not None and (high_cross_time is None or low_cross_time < high_cross_time):
            broke_low_first += 1
            after_low = after_open.loc[low_cross_time:]
            if (after_low["High"] >= or_high).any():
                broke_low_then_high += 1

    return total_days, broke_low_first, broke_low_then_high


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze opening range breaks")
    parser.add_argument("ticker", help="Ticker symbol")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--period", help="Period string for yfinance (e.g. 1mo, 6mo)")
    group.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--interval",
        default=None,
        help="Data interval (default determined automatically)",
    )
    args = parser.parse_args()

    if args.start and args.end:
        start = pd.to_datetime(args.start)
        end = pd.to_datetime(args.end)
        interval = args.interval or choose_yfinance_interval(start=start, end=end)
        df = fetch_intraday(args.ticker, start, end, interval=interval)
    else:
        interval = args.interval or choose_yfinance_interval(period=args.period)
        df, _ = fetch_stock(args.ticker, period=args.period, interval=interval)
        if df is None:
            df = pd.DataFrame()
        else:
            if "Datetime" in df.columns:
                idx = pd.DatetimeIndex(pd.to_datetime(df["Datetime"], errors="coerce"))
            else:
                idx = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            df.index = idx
            if "Datetime" in df.columns:
                df = df.drop(columns=["Datetime"])
            df.sort_index(inplace=True)

    total, low_first, low_then_high = analyze_open_range(df)

    print(f"Total days analyzed: {total}")
    print(f"Broke low before high: {low_first} ({(low_first/total*100 if total else 0):.2f}%)")
    print(f"Broke low then above high: {low_then_high} ({(low_then_high/total*100 if total else 0):.2f}%)")


if __name__ == "__main__":
    main()
