import argparse
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import time
import pandas as pd
import matplotlib.pyplot as plt

from fetch_stock import fetch_stock
from stock_functions import choose_yfinance_interval, period_to_start_end


@dataclass
class OpenRangeBreakTotals:
    """Collection of summary statistics from :func:`analyze_open_range`."""

    total_days: int = 0
    total_trades: int = 0
    total_profit: float = 0.0
    closed_higher_than_open: int = 0
    broke_low_first: int = 0
    broke_low_then_high: int = 0
    broke_high_first: int = 0
    broke_high_then_low: int = 0
    or_high_before_low: int = 0
    or_low_before_high: int = 0
    low_before_high_close_up: int = 0
    high_before_low_close_up: int = 0
    high_before_low_map: dict[pd.Timestamp, bool] = field(default_factory=dict)
    low_before_high_details: list[dict[str, float | str | pd.Timestamp]] = field(
        default_factory=list
    )


def expand_ticker_args(ticker_args: list[str]) -> list[str]:
    """Expand any portfolio references in ``ticker_args``.

    Tickers beginning with ``+`` are treated as portfolio names. The
    corresponding file in ``./portfolios`` is read and the contained
    tickers are added to the list.
    """
    expanded: list[str] = []
    for token in ticker_args:
        if token.startswith("+"):
            name = token[1:]
            path = Path("portfolios") / f"{name}.txt"
            if path.exists():
                expanded.extend(path.read_text().split())
            else:
                print(f"Portfolio file not found: {path}")
        else:
            expanded.append(token)
    return expanded


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
    df: pd.DataFrame,
    open_range_minutes: int = 30,
    *,
    loss_pct: float = 0.35,
    profit_pct: float = 1.0,
) -> OpenRangeBreakTotals:
    """Analyze opening range breaks for each trading day.

    ``open_range_minutes`` specifies how many minutes after 9:30am EST make up
    the opening range. ``loss_pct`` and ``profit_pct`` configure the
    intraday stop loss and profit target percentages used when simulating
    trades after the opening range.

    Returns an :class:`OpenRangeBreakTotals` instance summarizing statistics for
    each day analyzed. ``closed_higher_than_open`` counts the number of days the
    close finished above the open. ``or_high_before_low`` and
    ``or_low_before_high`` count the number of days the high or low of the
    opening range was reached first, respectively. ``low_before_high_close_up``
    counts the subset of ``or_low_before_high`` days where the day's close
    finished above the open. ``high_before_low_close_up`` does the same for
    ``or_high_before_low`` days. ``high_before_low_map`` maps each date to
    ``True`` if the day's break of the opening range high occurred before the
    break of the low. ``low_before_high_details`` contains dictionaries with
    ``date``, ``open``, ``or_low``, ``or_high`` and ``close`` for days where the
    OR low was broken before the high and the close finished above the open.
    """
    if df.empty:
        return OpenRangeBreakTotals()

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)

    totals = OpenRangeBreakTotals()

    open_end = (
        pd.Timestamp("09:30") + timedelta(minutes=open_range_minutes)
    ).strftime("%H:%M")
    after_settlement = (
        pd.Timestamp(open_end) + timedelta(minutes=5)
    ).strftime("%H:%M")


    for date, day_df in grouped:
        morning = day_df.between_time("09:30", open_end)
        near_closing = day_df.between_time(open_end,after_settlement)
        rest_of_day = day_df.between_time(after_settlement,"16:00")
        if morning.empty:
            continue
        or_high = morning["High"].max()
        or_low = morning["Low"].min()
        or_high_time = morning["High"].idxmax()
        or_low_time = morning["Low"].idxmin()
        open_price = morning.iloc[0]["Open"]
        close_price = day_df.iloc[-1]["Close"]
        after_or_price = near_closing.iloc[0]["Open"]


        print(
            f"    {pd.to_datetime(date)} - Open: {open_price:.2f},  Close: {close_price:.2f}, OR Low: {or_low:.2f}, "
            f"OR High: {or_high:.2f}, After OR Price: {after_or_price:.2f}, "
        )



        if close_price > open_price:
            totals.closed_higher_than_open += 1
        """ buy condition: 
            OR high before OR low and settlement close > open    
        """
        if or_high_time < or_low_time:
            if after_or_price > open_price * 1.002:
                buy = after_or_price
                outcome, sell = determine_gain_or_loss(
                    rest_of_day,
                    buy_price=buy,
                    loss_pct=loss_pct,
                    profit_pct=profit_pct,
                )
                profit = ((sell - buy) / buy) * 100
                totals.total_trades += 1
                totals.total_profit += profit
                totals.low_before_high_details.append(
                    {
                        "date": pd.to_datetime(date),
                        "open": float(open_price),
                        "close": float(close_price),
                        "or_low": float(or_low),
                        "or_high": float(or_high),
                        "after_OR": float(after_or_price),
                        "profit": float(profit),
                        "result": outcome,
                    }
                )
            totals.or_high_before_low += 1
            if close_price > open_price:
                totals.high_before_low_close_up += 1
        elif or_low_time < or_high_time:


            totals.or_low_before_high += 1
            if close_price > open_price:
                totals.low_before_high_close_up += 1
        after_open = day_df[day_df.index > morning.index[-1]]
        if after_open.empty:
            totals.total_days += 1
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

        totals.total_days += 1

        if high_cross_time is not None and (low_cross_time is None or high_cross_time < low_cross_time):
            totals.broke_high_first += 1
            after_high = after_open.loc[high_cross_time:]
            totals.high_before_low_map[pd.to_datetime(date)] = True
            if (after_high["Low"] <= or_low).any():
                totals.broke_high_then_low += 1
        else:
            if low_cross_time is not None:
                totals.broke_low_first += 1
                after_low = after_open.loc[low_cross_time:]
                if (after_low["High"] >= or_high).any():
                    totals.broke_low_then_high += 1
            totals.high_before_low_map[pd.to_datetime(date)] = False

    return totals


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


def determine_gain_or_loss(
    rest_of_day: pd.DataFrame,
    buy_price: float,
    loss_pct: float,
    profit_pct: float,
) -> tuple[str, float]:
    """Return the trade outcome and exit price.

    ``rest_of_day`` should contain intraday data from the time of entry until
    the end of the trading day with ``High`` and ``Low`` columns. ``buy_price``
    is the entry price. ``loss_pct`` and ``profit_pct`` are percentages that
    define the stop loss and profit target.

    The function returns a tuple ``(result, exit_price)`` where ``result`` is
    ``"profit"`` if the profit target is hit first, ``"loss"`` if the stop loss
    is triggered first and ``"close"`` if neither target is reached. The
    ``exit_price`` reflects the price at which the trade would close.
    """

    if rest_of_day.empty:
        return "close", buy_price

    stop_price = buy_price * (1 - loss_pct / 100)
    target_price = buy_price * (1 + profit_pct / 100)

    loss_hit = rest_of_day[rest_of_day["Low"] <= stop_price]
    profit_hit = rest_of_day[rest_of_day["High"] >= target_price]

    if loss_hit.empty and profit_hit.empty:
        return "close", float(rest_of_day.iloc[-1]["Close"])

    if not profit_hit.empty and (
        loss_hit.empty or profit_hit.index[0] < loss_hit.index[0]
    ):
        return "profit", target_price

    return "loss", stop_price


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze opening range breaks")
    parser.add_argument(
        "ticker",
        nargs="+",
        help="Ticker symbol or a list of symbols separated by spaces",
    )
    group = parser.add_mutually_exclusive_group(required=False)
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
    parser.add_argument(
        "--loss-pct",
        type=float,
        default=0.35,
        help="Stop loss percentage from entry price (default 0.35)",
    )
    parser.add_argument(
        "--profit-pct",
        type=float,
        default=1.0,
        help="Profit target percentage from entry price (default 1.0)",
    )
    args = parser.parse_args()

    tickers = expand_ticker_args(args.ticker)

    super_total_trades = 0
    super_total_profit = 0

    for ticker in tickers:
        if args.start:
            start = pd.to_datetime(args.start)
            end = pd.to_datetime(args.end) if args.end else start
        elif args.period:
            end = pd.to_datetime(args.end) if args.end else None
            start, end = period_to_start_end(args.period, end=end)
        else:
            now = pd.Timestamp.now(tz="US/Eastern")
            nine_thirty = pd.Timestamp("09:30", tz="US/Eastern").time()
            if now.time() < nine_thirty:
                start = (now - pd.Timedelta(days=1)).normalize()
                if start.dayofweek == 5:  # Saturday -> use previous Friday
                    start -= pd.Timedelta(days=1)
                elif start.dayofweek == 6:  # Sunday -> use previous Friday
                    start -= pd.Timedelta(days=2)
                end = start + pd.Timedelta(days=1)
            else:
                start = end = now.normalize()
        interval = args.interval or choose_yfinance_interval(start=start, end=end)
        df = fetch_intraday(ticker, start, end, interval=interval)

        # Calculate open range percentages for plotting
        or_pct = calculate_open_range_pct(df, open_range_minutes=args.range)

        results = analyze_open_range(
            df,
            open_range_minutes=args.range,
            loss_pct=args.loss_pct,
            profit_pct=args.profit_pct,
        )

        print(f"Results for {ticker}:")
        print(f"  Total days analyzed: {results.total_days}")
        print(f"  Total trades: {results.total_trades}")
        print(f"  Total profit: {results.total_profit}")
        print(
            f"  Days closed higher than open: {results.closed_higher_than_open} "
            f"({(results.closed_higher_than_open / results.total_days * 100 if results.total_days else 0):.2f}%)"
        )
        print(f"  Broke low before high: {results.broke_low_first} ({(results.broke_low_first / results.total_days * 100 if results.total_days else 0):.2f}%)")
        print(f"  Broke low then above high: {results.broke_low_then_high} ({(results.broke_low_then_high / results.total_days * 100 if results.total_days else 0):.2f}%)")
        print(f"  Broke high before low: {results.broke_high_first} ({(results.broke_high_first / results.total_days * 100 if results.total_days else 0):.2f}%)")
        print(f"  Broke high then low: {results.broke_high_then_low} ({(results.broke_high_then_low / results.total_days * 100 if results.total_days else 0):.2f}%)")
        print(f"  OR high before low: {results.or_high_before_low} ({(results.or_high_before_low / results.total_days * 100 if results.total_days else 0):.2f}%)")
        print(f"  OR low before high: {results.or_low_before_high} ({(results.or_low_before_high / results.total_days * 100 if results.total_days else 0):.2f}%)")
        print(f"  Close higher than open when OR low before high: {results.low_before_high_close_up} ({(results.low_before_high_close_up / results.or_low_before_high * 100 if results.or_low_before_high else 0):.2f}%)")
        print(
            f"  Close higher than open when OR high before low: {results.high_before_low_close_up} "
            f"({(results.high_before_low_close_up / results.or_high_before_low * 100 if results.or_high_before_low else 0):.2f}%)"
        )
        super_total_trades += results.total_trades
        super_total_profit += results.total_profit

        if results.low_before_high_details:
            print("  Days with close higher than open when OR low before high:")
            for item in results.low_before_high_details:
                date_str = item["date"].strftime("%Y-%m-%d")
                print(
                    f"    {date_str} - Open: {item['open']:.2f}, OR Low: {item['or_low']:.2f}, "
                    f"OR High: {item['or_high']:.2f}, Close: {item['close']:.2f}, After OR Price: {item['after_OR']:.2f}, "
                    f"Profit: {item['profit']:.2f} ({item['result']})"
                )
        time.sleep(0.1)

#        if not or_pct.empty:
#            ax = or_pct.plot(title=f"Opening Range % for {ticker}")
#            colors = [
#                "green" if results.high_before_low_map.get(date, False) else "red"
#                for date in or_pct.index
#            ]
#            ax.scatter(or_pct.index, or_pct.values, c=colors, s=50, zorder=3)
#            ax.set_xlabel("Date")
#            ax.set_ylabel("Open Range %")
#            ax.tick_params(axis="x", rotation=45)
#            plt.tight_layout()
#            plt.show()

    print("Total Trades:",super_total_trades)
    print("Total Profit:",super_total_profit)

if __name__ == "__main__":
    main()
