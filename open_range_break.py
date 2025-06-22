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


def analyze_open_range(
    df: pd.DataFrame, open_range_minutes: int = 30
) -> tuple[int, int, int, int, int, int, int, int, int, int, dict, list]:
    """Analyze opening range breaks for each trading day.

    ``open_range_minutes`` specifies how many minutes after 9:30am EST make up
    the opening range.

    Returns tuple of ``(total_days, closed_higher_than_open, broke_low_first,
    broke_low_then_high, broke_high_first, broke_high_then_low,
    or_high_before_low, or_low_before_high, low_before_high_close_up,
    high_before_low_close_up, high_before_low_map,
    low_before_high_close_up_details)`` where
    ``closed_higher_than_open`` counts the number of days the close finished
    above the open. ``or_high_before_low`` and ``or_low_before_high`` count the
    number of days
    the high or low of the opening range was reached first, respectively.
    ``low_before_high_close_up`` counts the subset of ``or_low_before_high`` days
    where the day's close finished above the open. ``high_before_low_close_up``
    does the same for ``or_high_before_low`` days. ``high_before_low_map`` maps
    each date to ``True`` if the day's break of the opening range high occurred
    before the break of the low. ``low_before_high_close_up_details`` contains
    dictionaries with ``date``, ``open``, ``or_low``, ``or_high`` and ``close``
    for days where the OR low was broken before the high and the close finished
    above the open.
    """
    if df.empty:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, []

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)

    total_days = 0
    closed_higher_than_open = 0
    broke_low_first = 0
    broke_low_then_high = 0
    broke_high_first = 0
    broke_high_then_low = 0
    or_high_before_low = 0
    or_low_before_high = 0
    low_before_high_close_up = 0
    high_before_low_close_up = 0
    high_before_low_map: dict[pd.Timestamp, bool] = {}
    low_before_high_close_up_details: list[dict[str, float | pd.Timestamp]] = []

    open_end = (
        pd.Timestamp("09:30") + timedelta(minutes=open_range_minutes)
    ).strftime("%H:%M")

    for date, day_df in grouped:
        morning = day_df.between_time("09:30", open_end)
        if morning.empty:
            continue
        or_high = morning["High"].max()
        or_low = morning["Low"].min()
        or_high_time = morning["High"].idxmax()
        or_low_time = morning["Low"].idxmin()
        open_price = morning.iloc[0]["Open"]
        close_price = day_df.iloc[-1]["Close"]
        if close_price > open_price:
            closed_higher_than_open += 1
        if or_high_time < or_low_time:
            or_high_before_low += 1
            if close_price > open_price:
                high_before_low_close_up += 1
        elif or_low_time < or_high_time:
            or_low_before_high += 1
            if close_price > open_price:
                low_before_high_close_up += 1
                low_before_high_close_up_details.append(
                    {
                        "date": pd.to_datetime(date),
                        "open": float(open_price),
                        "or_low": float(or_low),
                        "or_high": float(or_high),
                        "close": float(close_price),
                    }
                )
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
        closed_higher_than_open,
        broke_low_first,
        broke_low_then_high,
        broke_high_first,
        broke_high_then_low,
        or_high_before_low,
        or_low_before_high,
        low_before_high_close_up,
        high_before_low_close_up,
        high_before_low_map,
        low_before_high_close_up_details,
    )


def calculate_open_range_pct(
    df: pd.DataFrame, open_range_minutes: int = 30
) -> pd.Series:
    """Return a Series of opening range percentages indexed by date."""
    if df.empty:
        return pd.Series(dtype=float)

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)

    pct_values = {}
    open_end = (
        pd.Timestamp("09:30") + timedelta(minutes=open_range_minutes)
    ).strftime("%H:%M")
    for date, day_df in grouped:
        morning = day_df.between_time("09:30", open_end)
        if morning.empty:
            continue
        or_high = morning["High"].max()
        or_low = morning["Low"].min()
        open_price = morning.iloc[0]["Open"]
        pct_values[pd.to_datetime(date)] = (or_high - or_low) / open_price * 100

    return pd.Series(pct_values).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze opening range breaks")
    parser.add_argument(
        "ticker",
        nargs="+",
        help="Ticker symbol or a list of symbols separated by spaces",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--period", help="Period string for yfinance (e.g. 1mo, 6mo)")
    group.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--interval",
        default=None,
        help="Data interval (default determined automatically)",
    )
    parser.add_argument(
        "--range",
        type=int,
        default=30,
        help="Opening range in minutes (default 30)",
    )
    args = parser.parse_args()

    for ticker in args.ticker:
        if args.start and args.end:
            start = pd.to_datetime(args.start)
            end = pd.to_datetime(args.end)
            interval = args.interval or choose_yfinance_interval(start=start, end=end)
            df = fetch_intraday(ticker, start, end, interval=interval)
        else:
            end = pd.to_datetime(args.end) if args.end else None
            start, end = period_to_start_end(args.period, end=end)
            interval = args.interval or choose_yfinance_interval(start=start, end=end)
            df = fetch_intraday(ticker, start, end, interval=interval)

        # Calculate open range percentages for plotting
        or_pct = calculate_open_range_pct(df, open_range_minutes=args.range)

        (
            total_days,
            closed_higher_than_open,
            broke_low_first,
            broke_low_then_high,
            broke_high_first,
            broke_high_then_low,
            or_high_before_low,
            or_low_before_high,
            low_before_high_close_up,
            high_before_low_close_up,
            high_before_low_map,
            low_before_high_close_up_details,
        ) = analyze_open_range(df, open_range_minutes=args.range)

        print(f"Results for {ticker}:")
        print(f"  Total days analyzed: {total_days}")
        print(
            f"  Days closed higher than open: {closed_higher_than_open} "
            f"({(closed_higher_than_open / total_days * 100 if total_days else 0):.2f}%)"
        )
        print(f"  Broke low before high: {broke_low_first} ({(broke_low_first / total_days * 100 if total_days else 0):.2f}%)")
        print(f"  Broke low then above high: {broke_low_then_high} ({(broke_low_then_high / total_days * 100 if total_days else 0):.2f}%)")
        print(f"  Broke high before low: {broke_high_first} ({(broke_high_first / total_days * 100 if total_days else 0):.2f}%)")
        print(f"  Broke high then low: {broke_high_then_low} ({(broke_high_then_low / total_days * 100 if total_days else 0):.2f}%)")
        print(f"  OR high before low: {or_high_before_low} ({(or_high_before_low / total_days * 100 if total_days else 0):.2f}%)")
        print(f"  OR low before high: {or_low_before_high} ({(or_low_before_high / total_days * 100 if total_days else 0):.2f}%)")
        print(f"  Close higher than open when OR low before high: {low_before_high_close_up} ({(low_before_high_close_up / or_low_before_high * 100 if or_low_before_high else 0):.2f}%)")
        print(
            f"  Close higher than open when OR high before low: {high_before_low_close_up} "
            f"({(high_before_low_close_up / or_high_before_low * 100 if or_high_before_low else 0):.2f}%)"
        )
        if low_before_high_close_up_details:
            print("  Days with close higher than open when OR low before high:")
            for item in low_before_high_close_up_details:
                date_str = item["date"].strftime("%Y-%m-%d")
                print(
                    f"    {date_str} - Open: {item['open']:.2f}, OR Low: {item['or_low']:.2f}, "
                    f"OR High: {item['or_high']:.2f}, Close: {item['close']:.2f}"
                )

        if not or_pct.empty:
            ax = or_pct.plot(title=f"Opening Range % for {ticker}")
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
