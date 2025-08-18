import argparse

try:
    from tabulate import tabulate
except Exception:  # ImportError or other
    tabulate = None
from datetime import timedelta
from pathlib import Path

import time
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas.tseries.offsets import BDay

from stock_functions import round_numeric_cols

from stock_functions import choose_yfinance_interval, period_to_start_end
from open_range import OpenRangeAnalyzer
from portfolio_utils import expand_ticker_args, sanitize_ticker_string
from backtest_filters import fetch_daily_data, merge_indicator_data, passes_filters



def fetch_current_and_open_price(ticker: str) -> tuple[float, pd.Timestamp, float] | None:
    """Return the current price, timestamp, and today's open for ``ticker``."""
    data = yf.download(
        ticker, period="1d", interval="1m", auto_adjust=True, progress=False
    )
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index)
    if data.index.tz is None or data.index.tzinfo is None:
        data.index = data.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        data.index = data.index.tz_convert("US/Eastern")
    open_price = float(data["Open"].iloc[0])
    last_row = data.iloc[-1]
    current_price = float(last_row["Close"])
    price_time = data.index[-1]
    return current_price, price_time, open_price


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
        default="MO",
        help=(
            "Space-separated trade filters. Prefix with ! to invert. "
            "Available filters: MO (Mark > Open), OM (Open > Mark), ORM (Buy "
            "Price * 1.002 > Open Range High), GU (Open > Prev Close), "
            "GD (Open < Prev Close)"
        ),
    )
    parser.add_argument(
        "--filter-offset",
        type=float,
        default=1.0,
        help="Offset multiplier for filter comparison (default 1.0)",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        help="Stop analyzing after this many trades per ticker",
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=1.9,
        help="Minimum total profit to display details (default 1.9)",
    )
    parser.add_argument(
        "--console-out",
        default="",
        help=(
            "Space separated options to print to console. "
            "Use 'trades' to show all trades and 'tickers' for the per-ticker summary"
        ),
    )
    parser.add_argument(
        "--tickers",
        action="store_true",
        help="Print per-ticker summary to console in an ASCII table",
    )
    parser.add_argument(
        "--plot",
        choices=["daily"],
        help="Generate plots. 'daily' shows average profit and trade counts per day",
    )
    parser.add_argument(
        "--indicators",
        nargs="+",
        choices=[
            "price",
            "avg_vol",
            "dollar_vol",
            "atr_pct",
            "above_sma",
            "below_sma",
            "trend_slope",
            "nr7",
            "inside_2",
            "body_pct",
            "upper_wick",
            "lower_wick",
            "pullback_pct",
            "gap",
        ],
        help=(
            "Indicator filters to enable (default none). "
            "Example: --indicators price atr_pct gap"  # previously all enabled by default
        ),
    )
    # === Filtering flags ===
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=200.0)
    parser.add_argument("--min-avg-vol", dest="min_avg_vol", type=float, default=1_000_000)
    parser.add_argument("--min-dollar-vol", dest="min_dollar_vol", type=float, default=20_000_000)
    parser.add_argument("--min-atr-pct", type=float, default=1.0)
    parser.add_argument("--max-atr-pct", type=float, default=8.0)
    parser.add_argument("--above-sma", type=int, choices=[20, 50, 200], default=20)
    parser.add_argument("--below-sma", type=int, choices=[20, 50, 200], default=20)
    parser.add_argument("--trend-slope", type=float, default=0.0)
    parser.add_argument("--min-gap-pct", type=float, default=0.4)
    parser.add_argument("--body-pct-min", dest="body_pct_min", type=float, default=60.0)
    parser.add_argument("--upper-wick-max", dest="upper_wick_max", type=float, default=30.0)
    parser.add_argument("--lower-wick-max", dest="lower_wick_max", type=float, default=40.0)
    parser.add_argument("--pullback-pct-max", dest="pullback_pct_max", type=float, default=6.0)
    args = parser.parse_args()

    ticker_label = sanitize_ticker_string(args.ticker)
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

    chart_dir = Path("eldoradoCharts") / (
        f"backtest-{start.strftime('%m-%d-%Y')}-"
        f"{end.strftime('%m-%d-%Y')}-"
        f"{args.filter.replace(' ', '_')}"
    )
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_path = chart_dir / f"{ticker_label}-{args.range}.png"

    # Skip tickers that already have results saved under ./tickers
    ticker_root = Path("tickers")
    dir_suffix = (
        f"{start.strftime('%m-%d-%Y')}-"
        f"{end.strftime('%m-%d-%Y')}-"
        f"{args.filter.replace(' ', '_')}"
    )
    dest_dir = ticker_root / dir_suffix
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{ticker_label}-{args.range}.csv"
    existing_df: pd.DataFrame | None = None
#    if dest_file.is_file():
#        try:
#            existing_df = pd.read_csv(dest_file)
#            processed = set(existing_df.get("ticker", []).astype(str).str.upper())
#            tickers = [t for t in tickers if t.upper() not in processed]
#            if not tickers:
#                print(f"All tickers already processed in {dest_file}")
#        except Exception as e:
#            print(f"Failed to read existing ticker data {dest_file}: {e}")
#            existing_df = None

    for ticker in tickers:
        interval = args.interval or choose_yfinance_interval(start=start, end=end)

        analyzer = OpenRangeAnalyzer(
            interval=interval,
            open_range_minutes=args.range,
            loss_pct=args.loss_pct,
            profit_pct=args.profit_pct,
            filter=args.filter,
            filter_offset=args.filter_offset,
            max_trades=args.max_trades,
        )
        results, or_pct = analyzer.analyze_ticker(ticker, start, end)

        # Apply daily filters to each trade
        cache_tag = f"{start.date()}_{end.date()}"
        fetch_start = start - pd.Timedelta(days=5)
        fetch_end = end + pd.Timedelta(days=1)
        daily_df = fetch_daily_data(ticker, fetch_start, fetch_end, cache_tag, "eldorado")
        enabled_indicators = None
        if args.indicators:
            daily_df, have_ind = merge_indicator_data(daily_df, ticker)
            if have_ind:
                enabled_indicators = args.indicators
        filtered_details: list[dict[str, float | str | pd.Timestamp]] = []
        for item in results.trade_details:
            trade_date = pd.to_datetime(item.get("date")).date()
            exit_day = (pd.Timestamp(trade_date) + BDay(1)).date()
            idx = daily_df.index[daily_df["Date"].dt.date == exit_day]
            if len(idx) and (
                not enabled_indicators or passes_filters(daily_df, int(idx[0]), args, enabled_indicators)
            ):
                filtered_details.append(item)
        results.trade_details = filtered_details
        results.total_trades = len(filtered_details)
        results.total_profit = sum(d.get("profit", 0.0) for d in filtered_details)
        results.total_top_profit = sum(d.get("top_profit", 0.0) for d in filtered_details)

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
        if results.trade_details and results.total_profit > args.min_profit:
#            print(f"  Total days analyzed: {results.total_days}")
#            print(f"  Total trades: {results.total_trades}")
#            print(f"  Total profit: {results.total_profit}")
            for item in results.trade_details:
                date_str = item["date"].strftime("%Y-%m-%d")
#                print(
#                    f"    {date_str} - Open: {item['open']:.2f}, OR Low: {item['or_low']:.2f}, "
#                    f"OR High: {item['or_high']:.2f}, Close: {item['close']:.2f}, Buy Price: {item['buy_price']:.2f}, "
#                    f"Profit: {item['profit']:.2f} ({item['result']})"
#                )

        successes = sum(1 for d in results.trade_details if d["profit"] > 0)
        success_pct = (successes / results.total_trades * 100) if results.total_trades else 0
        minutes_list = [d["minutes"] for d in results.trade_details if d.get("minutes") is not None]
        avg_minutes = sum(minutes_list) / len(minutes_list) if minutes_list else 0
        ticker_rows.append(
            {
                "ticker": ticker,
                "total_trades": results.total_trades,
                "trade_success_pct": success_pct,
                "total_profit": results.total_profit,
                "total_top_profit": results.total_top_profit,
                "avg_trade_time": avg_minutes,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "range": args.range,
            }
        )

        for item in results.trade_details:
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

    daily_profit = None
    daily_top_profit = None
    daily_trades = None
    daily_trade_types = None
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        if args.plot == "daily":
            plot_df = trades_df.copy()
            if "date" in plot_df.columns:
                plot_df["date"] = pd.to_datetime(plot_df["date"]).dt.date
                if "profit" in plot_df.columns:
                    daily_profit = plot_df.groupby("date")["profit"].mean()
                if "top_profit" in plot_df.columns:
                    daily_top_profit = plot_df.groupby("date")["top_profit"].mean()
                daily_trades = plot_df.groupby("date").size()
                if "result" in plot_df.columns:
                    daily_trade_types = (
                        plot_df.groupby(["date", "result"]).size().unstack(fill_value=0)
                    )
                    daily_trade_types = daily_trade_types.reindex(
                        daily_profit.index, fill_value=0
                    )
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
        if "trades" in args.console_out.split():
            if tabulate:
                print(tabulate(trades_df, headers="keys", tablefmt="grid", showindex=False))
            else:
                print(trades_df.to_string(index=False))
        print(f"Trades saved to {trades_path}")

    avg_total_profit = super_total_profit / super_total_trades if super_total_trades else 0
    avg_total_top_profit = super_total_top_profit / super_total_trades if super_total_trades else 0

    if ticker_rows or existing_df is not None:
        combined_rows: list[dict[str, float | str]] = []
        if existing_df is not None:
            combined_rows.extend(existing_df.to_dict("records"))
        combined_rows.extend(ticker_rows)

        tickers_df = pd.DataFrame(combined_rows)
        dup_cols = ["ticker"]
        if "start_date" in tickers_df.columns:
            dup_cols.append("start_date")
        if "end_date" in tickers_df.columns:
            dup_cols.append("end_date")
        if "range" in tickers_df.columns:
            dup_cols.append("range")
        elif "open_range_minutes" in tickers_df.columns:
            dup_cols.append("open_range_minutes")
        tickers_df = tickers_df.drop_duplicates(subset=dup_cols, keep="last")
        raw_tickers_df = tickers_df.copy()

        surpass_df = (
            tickers_df[tickers_df["total_profit"] > args.min_profit]
            if args.min_profit is not None
            else tickers_df
        )

        surpass_by_profit = surpass_df.sort_values(by="total_profit", ascending=False)[
            "ticker"
        ].tolist()
        surpass_by_top_profit = surpass_df.sort_values(by="total_top_profit", ascending=False)[
            "ticker"
        ].tolist()
        surpass_by_success = surpass_df.sort_values(by="trade_success_pct", ascending=False)[
            "ticker"
        ].tolist()


        tickers_df = tickers_df.sort_values(by="total_profit", ascending=False)
        if args.min_profit is not None:
            tickers_df = tickers_df[tickers_df["total_profit"] > args.min_profit]


        for col in [
            "trade_success_pct",
            "total_profit",
            "total_top_profit",
            "avg_trade_time",
        ]:
            if col in tickers_df.columns:
                tickers_df[col] = tickers_df[col].map(lambda x: f"{x:.2f}")


        tickers_df = round_numeric_cols(tickers_df)
        tickers_df.to_csv(tickers_path, index=False)

        dest_dir.mkdir(parents=True, exist_ok=True)
        combined = round_numeric_cols(raw_tickers_df)
        combined.to_csv(dest_file, index=False)

        if args.tickers or "tickers" in args.console_out.split():
            if tabulate:
                print(tabulate(tickers_df, headers="keys", tablefmt="grid", showindex=False))
            else:
                print(tickers_df.to_string(index=False))
        print(f"Ticker summary saved to {tickers_path}")

        if surpass_by_profit:
            print(
                "Tickers surpassing min profit by total profit:",
                " ".join(surpass_by_profit),
            )
        if surpass_by_top_profit:
            print(
                "Tickers surpassing min profit by total top profit:",
                " ".join(surpass_by_top_profit),
            )
        if surpass_by_success:
            print(
                "Tickers surpassing min profit by success percentage:",
                " ".join(surpass_by_success),
            )

        # If analysis ended on the last complete trading day, check for today's buys
        today_buys: list[dict[str, float | str | pd.Timestamp]] = []
        end_date = end.tz_convert("US/Eastern").date() if end.tzinfo else end.date()
        start_date = start.tz_convert("US/Eastern").date() if start.tzinfo else start.date()
        last_day = end_date if end_date == start_date else (pd.Timestamp(end_date) - pd.Timedelta(days=1)).date()

        now_eastern = pd.Timestamp.now(tz="US/Eastern")
        market_close = pd.Timestamp("16:00", tz="US/Eastern").time()
        if now_eastern.dayofweek >= 5:
            last_complete_day = (now_eastern - BDay(1)).date()
        elif now_eastern.time() >= market_close:
            last_complete_day = now_eastern.date()
        else:
            last_complete_day = (now_eastern - BDay(1)).date()
        print(last_day,last_complete_day,end_date)
        if end_date == last_complete_day:
            top_tickers = sorted(
                set(surpass_by_profit) | set(surpass_by_top_profit) | set(surpass_by_success)
            )
            for t in top_tickers:
                result = fetch_current_and_open_price(t)
                if result is None:
                    continue
                current_price, price_time, open_price = result
                if current_price > open_price:
                    today_buys.append(
                        {
                            "ticker": t.upper(),
                            "current_price": current_price,
                            "price_time": price_time,
                            "open_price": open_price,
                        }
                    )
        if today_buys:
            today_df = pd.DataFrame(today_buys)
            stats_df = raw_tickers_df.copy()
            stats_df["ticker"] = stats_df["ticker"].astype(str).str.upper()
            today_df["ticker"] = today_df["ticker"].astype(str).str.upper()
            today_df = today_df.merge(stats_df, on="ticker", how="left")
            if "price_time" in today_df.columns:
                today_df["price_time"] = pd.to_datetime(today_df["price_time"]).dt.strftime("%H:%M")
            for col in ["current_price", "open_price"]:
                if col in today_df.columns:
                    today_df[col] = today_df[col].map(lambda x: f"{x:.2f}")
            for col in [
                "trade_success_pct",
                "total_profit",
                "total_top_profit",
                "avg_trade_time",
            ]:
                if col in today_df.columns:
                    today_df[col] = today_df[col].map(lambda x: f"{x:.2f}")
            cols = [
                "ticker",
                "current_price",
                "open_price",
                "price_time",
                "total_trades",
                "trade_success_pct",
                "total_profit",
                "total_top_profit",
                "avg_trade_time",
                "start_date",
                "end_date",
                "range",
            ]
            today_df = today_df[[c for c in cols if c in today_df.columns]]
            print("Today's Buys:")
            if tabulate:
                print(tabulate(today_df, headers="keys", tablefmt="grid", showindex=False))
            else:
                print(today_df.to_string(index=False))

    print("Total Trades:", super_total_trades)
    print("Total Profit:", f"{super_total_profit:.3f}")
    print("Total Top Profit:", f"{super_total_top_profit:.3f}")
    print("Avg Total Profit:",f"{avg_total_profit:.4f}")
    print("Avg Total Top Profit:",f"{avg_total_top_profit:.4f}")

    if (
        args.plot == "daily"
        and daily_profit is not None
        and not daily_profit.empty
        and daily_trades is not None
        and daily_trade_types is not None
    ):
        x = range(len(daily_profit))

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_profit = "tab:blue"
        line_profit = ax1.plot(
            x,
            daily_profit.values,
            marker="o",
            color=color_profit,
            linewidth=3,
            label="Avg Profit",
        )[0]

        if daily_top_profit is not None:
            color_top = "tab:purple"
            line_top = ax1.plot(
                x,
                daily_top_profit.reindex(daily_profit.index).values,
                marker="o",
                linestyle="--",
                color=color_top,
                linewidth=2,
                label="Avg Top Profit",
            )[0]
        else:
            line_top = None

        ax1.set_ylabel("Average Profit (%)", color=color_profit)
        ax1.tick_params(axis="y", labelcolor=color_profit)
        ax1.set_xlabel("Date")
        ax1.set_title("Daily Profit and Trades")
        ax1.grid(True)

        ax2 = ax1.twinx()
        profit_counts = daily_trade_types.get("profit", pd.Series(0, index=daily_profit.index))
        close_counts = daily_trade_types.get("close", pd.Series(0, index=daily_profit.index))
        loss_counts = daily_trade_types.get("loss", pd.Series(0, index=daily_profit.index))
        bar1 = ax2.bar(x, profit_counts, color="green", alpha=0.6, label="Profit")
        bar2 = ax2.bar(x, close_counts, bottom=profit_counts, color="blue", alpha=0.6, label="Close")
        ax2.bar(
            x,
            loss_counts,
            bottom=profit_counts + close_counts,
            color="red",
            alpha=0.6,
            label="Loss",
        )
        ax2.set_ylabel("Trades")
        ax2.tick_params(axis="y")
        ax2.legend(loc="upper right")

        lines = [line_profit]
        if line_top is not None:
            lines.append(line_top)
        labels = [l.get_label() for l in lines]

        summary_text = (
            f"Total Trades: {super_total_trades}\n"
            f"Total Profit: {super_total_profit:.3f}\n"
            f"Total Top Profit: {super_total_top_profit:.3f}\n"
            f"Avg Total Profit: {avg_total_profit:.4f}\n"
            f"Avg Total Top Profit: {avg_total_top_profit:.4f}"
        )

        ax1.legend(lines, labels, loc="upper left", title=summary_text)

        ax1.set_xticks(list(x))
        ax1.set_xticklabels(
            [d.strftime("%Y-%m-%d") for d in daily_profit.index], rotation=45
        )

        fig.tight_layout()
        plt.savefig(chart_path)
        plt.show()

if __name__ == "__main__":
    main()
