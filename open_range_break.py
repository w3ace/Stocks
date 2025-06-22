import argparse
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt

from fetch_stock import fetch_stock
from stock_functions import choose_yfinance_interval, period_to_start_end


def fetch_intraday(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str = "5m"
) -> pd.DataFrame:
    """Fetch intraday data for ``ticker`` using :func:`fetch_stock`.

    ``start`` and ``end`` may be timezone-naive or aware. They are localized to
    UTC so that comparisons against the fetched data's index work reliably.
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    data = fetch_stock(ticker, start_date=start, end_date=end, interval=interval)
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


def analyze_open_range(df: pd.DataFrame) -> tuple[int, int, int, int, int, dict]:
    """Analyze opening range breaks for each trading day.

    Returns tuple of ``(total_days, broke_low_first, broke_low_then_high,
    broke_high_first, broke_high_then_low, high_before_low_map)`` where
    ``high_before_low_map`` maps each date to ``True`` if the day's break of the
    opening range high occurred before the break of the low.
    """
    if df.empty:
        return 0, 0, 0, 0, 0, {}

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)

    total_days = 0
    broke_low_first = 0
    broke_low_then_high = 0
    broke_high_first = 0
    broke_high_then_low = 0
    high_before_low_map: dict[pd.Timestamp, bool] = {}

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

        if high_cross_time is not None and (low_cross_time is None or high_cross_time < low_cross_time):
            broke_high_first += 1
            after_high = after_open.loc[high_cross_time:]
            high_before_low_map[pd.to_datetime(date)] = True
            if (after_high["Low"] <= or_low).any():
                broke_high_then_low += 1
        else:
            if low_cross_time is not None:
                broke_low_first += 1
                after_low = after_open.loc[low_cross_time:]
                if (after_low["High"] >= or_high).any():
                    broke_low_then_high += 1
            high_before_low_map[pd.to_datetime(date)] = False

    return (
        total_days,
        broke_low_first,
        broke_low_then_high,
        broke_high_first,
        broke_high_then_low,
        high_before_low_map,
    )


def calculate_open_range_pct(df: pd.DataFrame) -> pd.Series:
    """Return a Series of opening range percentages indexed by date."""
    if df.empty:
        return pd.Series(dtype=float)

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)

    pct_values = {}
    for date, day_df in grouped:
        morning = day_df.between_time("09:30", "10:00")
        if morning.empty:
            continue
        or_high = morning["High"].max()
        or_low = morning["Low"].min()
        open_price = morning.iloc[0]["Open"]
        pct_values[pd.to_datetime(date)] = (or_high - or_low) / open_price * 100

    return pd.Series(pct_values).sort_index()


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
        end = pd.to_datetime(args.end) if args.end else None
        start, end = period_to_start_end(args.period, end=end)
        interval = args.interval or choose_yfinance_interval(start=start, end=end)
        df = fetch_intraday(args.ticker, start, end, interval=interval)

    # Calculate open range percentages for plotting
    or_pct = calculate_open_range_pct(df)

    (
        total,
        low_first,
        low_then_high,
        high_first,
        high_then_low,
        high_before_low_map,
    ) = analyze_open_range(df)

    print(f"Total days analyzed: {total}")
    print(f"Broke low before high: {low_first} ({(low_first/total*100 if total else 0):.2f}%)")
    print(
        f"Broke low then above high: {low_then_high} ({(low_then_high/total*100 if total else 0):.2f}%)"
    )
    print(f"Broke high before low: {high_first} ({(high_first/total*100 if total else 0):.2f}%)")
    print(
        f"Broke high then low: {high_then_low} ({(high_then_low/total*100 if total else 0):.2f}%)"
    )

    if not or_pct.empty:
        ax = or_pct.plot(title=f"Opening Range % for {args.ticker}")
        colors = [
            "green" if high_before_low_map.get(date, False) else "red"
            for date in or_pct.index
        ]
        ax.scatter(or_pct.index, or_pct.values, c=colors, s=50, zorder=3)
        ax.set_xlabel("Date")
        ax.set_ylabel("Open Range %")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
