import argparse

try:
    from tabulate import tabulate
except Exception:  # ImportError or other
    tabulate = None
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
    total_top_profit: float = 0.0
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
            path = Path("portfolios") / f"{name}"
            if path.exists():
                expanded.extend(path.read_text().split())
            else:
                print(f"Portfolio file not found: {path}")
        else:
            expanded.append(token)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(expanded))


def fetch_intraday(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str = "5m"
) -> pd.DataFrame:
    """Fetch intraday data for ``ticker`` using :func:`fetch_stock`.

    ``start`` and ``end`` may be timezone-naive or aware. They are localized to
    UTC so that comparisons against the fetched data's index work reliably. The
    ``start`` time is always normalized to 9:30am US/Eastern before converting to
    UTC so that yfinance downloads begin at the market open and the cache key
    remains consistent.
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # Normalize the start time to 9:30am US/Eastern for yfinance and caching
    start_est = start.tz_localize("US/Eastern") if start.tzinfo is None else start.tz_convert("US/Eastern")
    start_est = start_est.normalize() + pd.Timedelta(hours=9, minutes=30)
    start = start_est.tz_convert("UTC")

    # End time at 4:00pm US/Eastern if no explicit time provided
    end_est = end.tz_localize("US/Eastern") if end.tzinfo is None else end.tz_convert("US/Eastern")
    if end_est.time() == pd.Timestamp("00:00", tz="US/Eastern").time():
        end_est = end_est.normalize() + pd.Timedelta(hours=16)
    end = end_est.tz_convert("UTC")

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
    filter: str = "MO",
    filter_offset: float = 1.0,
) -> OpenRangeBreakTotals:
    """Analyze opening range breaks for each trading day.

    ``open_range_minutes`` specifies how many minutes after 9:30am EST make up
    the opening range. ``loss_pct`` and ``profit_pct`` configure the
    intraday stop loss and profit target percentages used when simulating
    trades after the opening range. ``filter`` determines how the closing
    price of the opening range ("Mark") is compared to the day's open when
    deciding whether to take a trade. ``filter_offset`` is a multiplier used in
    that comparison.

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
        near_closing = day_df.between_time(open_end, after_settlement)
        rest_of_day = day_df.between_time(after_settlement, "16:00")
        if morning.empty:
            continue
        or_high = morning["High"].max()
        or_low = morning["Low"].min()
        or_high_time = morning["High"].idxmax()
        or_low_time = morning["Low"].idxmin()
        open_price = morning.iloc[0]["Open"]
        close_price = day_df.iloc[-1]["Close"]
        if near_closing.empty:
            # Skip the day if there is no data immediately after the opening
            # range. Attempting to access ``near_closing.iloc[0]`` would raise
            # ``IndexError`` otherwise.
            continue
        after_or_price = near_closing.iloc[0]["Open"]
        after_or_time = near_closing.index[0]


#      `print(
#            f"    {pd.to_datetime(date)} - Open: {open_price:.2f},  Close: {close_price:.2f}, OR Low: {or_low:.2f}, "
#            f"OR High: {or_high:.2f}, After OR Price: {after_or_price:.2f}, "
#        )



        if close_price > open_price:
            totals.closed_higher_than_open += 1
        """ buy condition: 
            OR high before OR low and settlement close > open    
        """
        should_buy = False
        if filter.upper() == "MO":
            should_buy = after_or_price > open_price * filter_offset
        elif filter.upper() == "OM":
            should_buy = open_price > after_or_price * filter_offset

        if should_buy:
            buy = after_or_price
            outcome, sell, sell_time, top_price = determine_gain_or_loss(
                rest_of_day,
                buy_price=buy,
                loss_pct=loss_pct,
                profit_pct=profit_pct,
            )
            profit = ((sell - buy) / buy) * 100
            top_profit_pct = ((top_price - buy) / buy) * 100
            minutes = (
                (sell_time - after_or_time).total_seconds() / 60
                if pd.notna(sell_time)
                else None
            )
            stop_price = buy * (1 - loss_pct / 100)
            target_price = buy * (1 + profit_pct / 100)
            totals.total_trades += 1
            totals.total_profit += profit
            totals.total_top_profit += top_profit_pct
            totals.low_before_high_details.append(
                {
                    "date": pd.to_datetime(date),
                    "time": after_or_time,
                    "open": float(open_price),
                    "close": float(close_price),
                    "or_low": float(or_low),
                    "or_high": float(or_high),
                    "buy_price": float(buy),
                    "stop_price": float(stop_price),
                    "profit_price": float(target_price),
                    "profit": float(profit),
                    "top_profit": float(top_profit_pct),
                    "result": outcome,
                    "buy_time": after_or_time,
                    "sell_time": sell_time,
                    "minutes": minutes,
                }
            )
        if or_high_time < or_low_time:
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


def open_range_break(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    interval: str = "5m",
    open_range_minutes: int = 30,
    loss_pct: float = 0.35,
    profit_pct: float = 1.0,
    filter: str = "MO",
    filter_offset: float = 1.0,
) -> tuple[OpenRangeBreakTotals, pd.Series]:
    """Fetch data for ``ticker`` and analyze opening range breaks.

    Returns the :class:`OpenRangeBreakTotals` along with the opening range
    percentages for plotting. ``filter`` and ``filter_offset`` are passed
    through to :func:`analyze_open_range`.
    """

    df = fetch_intraday(ticker, start, end, interval=interval)
    or_pct = calculate_open_range_pct(df, open_range_minutes=open_range_minutes)
    results = analyze_open_range(
        df,
        open_range_minutes=open_range_minutes,
        loss_pct=loss_pct,
        profit_pct=profit_pct,
        filter=filter,
        filter_offset=filter_offset,
    )
    return results, or_pct


def determine_gain_or_loss(
    rest_of_day: pd.DataFrame,
    buy_price: float,
    loss_pct: float,
    profit_pct: float,
) -> tuple[str, float, pd.Timestamp, float]:
    """Return the trade outcome, exit price, time and top price.

    ``rest_of_day`` should contain intraday data from the time of entry until
    the end of the trading day with ``High`` and ``Low`` columns. ``buy_price``
    is the entry price. ``loss_pct`` and ``profit_pct`` are percentages that
    define the stop loss and profit target.

    The function returns a tuple ``(result, exit_price, exit_time, top_price)``
    where ``result`` is ``"profit"`` if the profit target is hit first,
    ``"loss"`` if the stop loss is triggered first and ``"close"`` if neither
    target is reached. ``exit_price`` is the price at which the trade would
    close, ``exit_time`` is the timestamp of that event and ``top_price`` is the
    highest price reached before the trade exited.
    """

    if rest_of_day.empty:
        return "close", buy_price, pd.NaT, buy_price

    stop_price = buy_price * (1 - loss_pct / 100)
    target_price = buy_price * (1 + profit_pct / 100)

    loss_hit = rest_of_day[rest_of_day["Low"] <= stop_price]
    profit_hit = rest_of_day[rest_of_day["High"] >= target_price]

    if loss_hit.empty and profit_hit.empty:
        exit_time = rest_of_day.index[-1]
        exit_price = float(rest_of_day.iloc[-1]["Close"])
        top_price = float(rest_of_day["High"].max())
        return "close", exit_price, exit_time, top_price

    first_loss_time = loss_hit.index[0] if not loss_hit.empty else None
    first_profit_time = profit_hit.index[0] if not profit_hit.empty else None

    if first_profit_time is not None and (
        first_loss_time is None or first_profit_time < first_loss_time
    ):
        exit_time = first_profit_time
        exit_price = target_price
        subset = rest_of_day.loc[:exit_time]
        top_price = float(subset["High"].max())
        return "profit", exit_price, exit_time, top_price
    if first_loss_time is not None:
        exit_time = first_loss_time
        exit_price = stop_price
        subset = rest_of_day.loc[:exit_time]
        top_price = float(subset["High"].max())
        return "loss", exit_price, exit_time, top_price

    # Should not reach here but return close at end of day as fallback
    exit_time = rest_of_day.index[-1]
    exit_price = float(rest_of_day.iloc[-1]["Close"])
    top_price = float(rest_of_day.loc[:exit_time]["High"].max())
    return "close", exit_price, exit_time, top_price


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
    parser.add_argument(
        "--filter",
        choices=["MO", "OM"],
        default="MO",
        help="Trade filter: MO for Mark > Open or OM for Open > Mark",
    )
    parser.add_argument(
        "--filter-offset",
        type=float,
        default=1.0,
        help="Offset multiplier for filter comparison (default 1.0)",
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=1.9,
        help="Minimum total profit to display details (default 1.9)",
    )
    parser.add_argument(
        "--output-trades",
        action="store_true",
        help="Print all trades to console in an ASCII table",
    )
    parser.add_argument(
        "--tickers",
        action="store_true",
        help="Print per-ticker summary to console in an ASCII table",
    )
    parser.add_argument(
        "--output-tickers",
        action="store_true",
        help="Output per-ticker metrics table to console",
    )
    args = parser.parse_args()

    tickers = expand_ticker_args(args.ticker)

    super_total_trades = 0
    super_total_profit = 0
    super_total_top_profit = 0

    all_trades: list[dict[str, float | str | pd.Timestamp]] = []
    ticker_rows: list[dict[str, float | str]] = []
    surpass_tickers: list[str] = []

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now(tz="US/Eastern").strftime("%Y%m%d_%H%M%S")

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

        results, or_pct = open_range_break(
            ticker,
            start,
            end,
            interval=interval,
            open_range_minutes=args.range,
            loss_pct=args.loss_pct,
            profit_pct=args.profit_pct,
            filter=args.filter,
            filter_offset=args.filter_offset,
        )

        """
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
        """


     #   print(f"Results for {ticker}:")

        super_total_trades += results.total_trades
        super_total_profit += results.total_profit
        super_total_top_profit += results.total_top_profit

        if results.total_profit > args.min_profit:
            surpass_tickers.append(ticker)
        if results.low_before_high_details and results.total_profit > args.min_profit:
#            print(f"  Total days analyzed: {results.total_days}")
#            print(f"  Total trades: {results.total_trades}")
#            print(f"  Total profit: {results.total_profit}")
            for item in results.low_before_high_details:
                date_str = item["date"].strftime("%Y-%m-%d")
#                print(
#                    f"    {date_str} - Open: {item['open']:.2f}, OR Low: {item['or_low']:.2f}, "
#                    f"OR High: {item['or_high']:.2f}, Close: {item['close']:.2f}, Buy Price: {item['buy_price']:.2f}, "
#                    f"Profit: {item['profit']:.2f} ({item['result']})"
#                )

        successes = sum(1 for d in results.low_before_high_details if d["profit"] > 0)
        success_pct = (successes / results.total_trades * 100) if results.total_trades else 0
        minutes_list = [d["minutes"] for d in results.low_before_high_details if d.get("minutes") is not None]
        avg_minutes = sum(minutes_list) / len(minutes_list) if minutes_list else 0
        ticker_rows.append(
            {
                "ticker": ticker,
                "total_trades": results.total_trades,
                "trade_success_pct": success_pct,
                "total_profit": results.total_profit,
                "total_top_profit": results.total_top_profit,
                "avg_trade_time": avg_minutes,
            }
        )

        for item in results.low_before_high_details:
            trade = item.copy()
            trade["ticker"] = ticker
            all_trades.append(trade)

  #      time.sleep(0.1)

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

    trades_path = output_dir / f"{timestamp}_trades.csv"
    tickers_path = output_dir / f"{timestamp}_tickers.csv"

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        desired_cols = [
            "date",
            "time",
            "ticker",
            "open",
            "close",
            "buy_price",
            "stop_price",
            "profit_price",
            "top_profit",
            "profit",
            "buy_time",
            "sell_time",
            "result",
            "minutes",
        ]
        trades_df = trades_df[[c for c in desired_cols if c in trades_df.columns]]

        if "date" in trades_df.columns:
            trades_df["date"] = pd.to_datetime(trades_df["date"]).dt.strftime("%Y-%m-%d")

        if "time" in trades_df.columns:
            trades_df = trades_df.drop(columns=["time"])

        for col in ["open", "close", "buy_price", "stop_price", "profit_price"]:
            if col in trades_df.columns:
                trades_df[col] = trades_df[col].map(lambda x: f"${x:,.2f}")

        for col in ["profit", "top_profit"]:
            if col in trades_df.columns:
                trades_df[col] = trades_df[col].map(lambda x: f"{x:.2f}")

        for col in ["buy_time", "sell_time"]:
            if col in trades_df.columns:
                trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime("%H:%M")

        if "result" in trades_df.columns:
            trades_df = trades_df.rename(columns={"result": "profit_or_loss"})

        trades_df.to_csv(trades_path, index=False)
        if args.output_trades:
            if tabulate:
                print(tabulate(trades_df, headers="keys", tablefmt="grid", showindex=False))
            else:
                print(trades_df.to_string(index=False))
        print(f"Trades saved to {trades_path}")

    if ticker_rows:
        tickers_df = pd.DataFrame(ticker_rows)

        # Order by total profit descending and apply --min-profit filter
        tickers_df = tickers_df.sort_values(by="total_profit", ascending=False)
        if args.min_profit is not None:
            tickers_df = tickers_df[tickers_df["total_profit"] > args.min_profit]

        for col in ["trade_success_pct", "total_profit", "total_top_profit", "avg_trade_time"]:
            if col in tickers_df.columns:
                tickers_df[col] = tickers_df[col].map(lambda x: f"{x:.2f}")

        tickers_df.to_csv(tickers_path, index=False)
        if args.tickers or args.output_tickers:
            if tabulate:
                print(tabulate(tickers_df, headers="keys", tablefmt="grid", showindex=False))
            else:
                print(tickers_df.to_string(index=False))
        print(f"Ticker summary saved to {tickers_path}")

    if surpass_tickers:
        print("Tickers surpassing min profit:", " ".join(surpass_tickers))

    print("Total Trades:", super_total_trades)
    print("Total Profit:", super_total_profit)
    print("Total Top Profit:", super_total_top_profit)

if __name__ == "__main__":
    main()
