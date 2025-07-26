import argparse
import subprocess
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Basic NYSE holiday calendar implemented with pandas.tseries.holiday so we
# can skip market holidays without the external pandas_market_calendars
# dependency.
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    nearest_workday,
    USMartinLutherKingJr,
    USPresidentsDay,
    GoodFriday,
    USMemorialDay,
    USLaborDay,
    USThanksgivingDay,
)


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Simplified NYSE holiday calendar."""

    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


nyse_calendar = NYSEHolidayCalendar()


def is_trading_day(day: date) -> bool:
    """Return True if *day* is a NYSE trading day."""

    if day.weekday() >= 5:  # Saturday/Sunday
        return False
    holidays = nyse_calendar.holidays(start=day, end=day)
    return pd.Timestamp(day) not in holidays


def plot_daily_results(df: pd.DataFrame) -> None:
    """Plot daily average profit and trade counts similar to backtest."""

    x = range(len(df))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_profit = "tab:blue"
    line_profit = ax1.plot(
        x,
        df["avg_profit"].values,
        marker="o",
        color=color_profit,
        linewidth=3,
        label="Avg Profit",
    )[0]

    if "avg_top_profit" in df.columns:
        color_top = "tab:purple"
        line_top = ax1.plot(
            x,
            df["avg_top_profit"].values,
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
    bar1 = ax2.bar(
        x,
        df["profit_count"].values,
        color="green",
        alpha=0.6,
        label="Profit",
    )
    bar2 = ax2.bar(
        x,
        df["close_count"].values,
        bottom=df["profit_count"].values,
        color="blue",
        alpha=0.6,
        label="Close",
    )
    ax2.bar(
        x,
        df["loss_count"].values,
        bottom=df["profit_count"].values + df["close_count"].values,
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
    ax1.legend(lines, labels, loc="upper left")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels([d.strftime("%Y-%m-%d") for d in df.index], rotation=45)

    fig.tight_layout()
    plt.show()


def run_backtest(arguments: list[str]) -> tuple[Path, Path]:
    """Run backtest.py with the given arguments and return output csv paths."""

    result = subprocess.run(
        ["python", "backtest.py", *arguments], capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("backtest.py failed")

    out_dir = Path("output")
    tickers = sorted(
        out_dir.glob("*_tickers.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    trades = sorted(
        out_dir.glob("*_trades.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not tickers or not trades:
        raise FileNotFoundError("Output csv files not found")

    return tickers[0], trades[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive backtest runner")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--range",
        type=int,
        default=30,
        help="Opening range in minutes (default 30)",
    )
    parser.add_argument(
        "--ticker-list",
        nargs="+",
        default=["AAPL NVDA GOOGL AMZN"],
        help="Tickers or portfolio names used for lookback selection",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=21,
        help="Number of days to test for ticker selection (default 21)",
    )
    parser.add_argument(
        "--loss-pct",
        type=float,
        default=0.35,
        help="Stop loss percentage (default 0.35)",
    )
    parser.add_argument(
        "--profit-pct",
        type=float,
        default=1.05,
        help="Profit target percentage (default 1.05)",
    )
    parser.add_argument(
        "--filter",
        default="MO",
        help=(
            "Space-separated trade filters. Prefix with ! to invert. "
            "Available filters: MO, OM, ORM"
        ),
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        help="Stop analyzing after this many trades per ticker",
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=-1.0,
        help="Minimum total profit to include ticker (default -1)",
    )
    parser.add_argument(
        "--console-out",
        default="",
        help="Space separated options to print to console (tickers, trades)",
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    current = start_date
    total_trades = 0
    total_profit = 0.0
    total_top_profit = 0.0
    day_count = 0
    ticker_history: list[list[str]] = []
    daily_stats: list[dict[str, float | int | datetime]] = []

    while current <= end_date:
        if not is_trading_day(current):
            # Skip weekends and NYSE holidays
            current += timedelta(days=1)
            continue
        lookback_start = current - timedelta(days=args.sample)
        lookback_end = current - timedelta(days=1)

        # Run backtest on the lookback period to select tickers
        csv_path, _ = run_backtest([
            "--end",
            lookback_end.strftime("%Y-%m-%d"),
            "--start",
            lookback_start.strftime("%Y-%m-%d"),
            "--range",
            str(args.range),
            "--loss-pct",
            str(args.loss_pct),
            "--profit-pct",
            str(args.profit_pct),
            "--filter",
            args.filter,
            "--min-profit",
            str(args.min_profit),
            *args.ticker_list,
            *( ["--console-out", args.console_out] if args.console_out else [] ),
        ])

        df = pd.read_csv(csv_path)

        tickers_top_profit = (
            df.sort_values(by="total_top_profit", ascending=False)["ticker"].head(6).tolist()
        )
        tickers_profit = (
            df.sort_values(by="total_profit", ascending=False)["ticker"].head(6).tolist()
        )

        success_col = (
            "trade_success_rate"
            if "trade_success_rate" in df.columns
            else "trade_success_pct" if "trade_success_pct" in df.columns
            else None
        )
        tickers_success = (
            df.sort_values(by=success_col, ascending=False)["ticker"].head(6).tolist()
            if success_col
            else []
        )

        tickers: list[str] = []
        for t in tickers_profit + tickers_top_profit + tickers_success:
            if t not in tickers:
                tickers.append(t)

        if not tickers:
            current += timedelta(days=1)
            continue

        ticker_history.append(tickers)

        # Run backtest for the current day using selected tickers
        result_csv, trades_csv = run_backtest([
            "--end",
            current.strftime("%Y-%m-%d"),
            "--start",
            current.strftime("%Y-%m-%d"),
            "--range",
            str(args.range),
            "--loss-pct",
            str(args.loss_pct),
            "--profit-pct",
            str(args.profit_pct),
            "--filter",
            str(args.filter),
            "--min-profit",
            str(args.min_profit),
            *tickers,
            *( ["--console-out", args.console_out] if args.console_out else [] ),
        ])

        result_df = pd.read_csv(result_csv)
        trades_df = pd.read_csv(trades_csv)

        total_trades += result_df["total_trades"].sum()
        total_profit += result_df["total_profit"].sum()
        total_top_profit += result_df["total_top_profit"].sum()
        day_count += 1

        avg_profit_day = (result_df["total_profit"].sum() / result_df["total_trades"].sum()) if result_df["total_trades"].sum() else 0.0
        avg_top_profit_day = (result_df["total_top_profit"].sum() / result_df["total_trades"].sum()) if result_df["total_trades"].sum()else 0.0
        
        if "profit_or_loss" in trades_df.columns:
            counts = trades_df["profit_or_loss"].value_counts()
        elif "result" in trades_df.columns:
            counts = trades_df["result"].value_counts()
        else:
            counts = pd.Series(dtype=int)

        if not counts.empty:
            counts.index = counts.index.str.lower()
        if(result_df["total_trades"].sum()):
            daily_stats.append(
                {
                    "date": current,
                    "avg_profit": avg_profit_day,
                    "avg_top_profit": avg_top_profit_day,
                    "profit_count": int(counts.get("profit", 0)),
                    "close_count": int(counts.get("close", 0)),
                    "loss_count": int(counts.get("loss", 0)),
                }
            )

        current += timedelta(days=1)

    avg_profit = total_profit / total_trades if total_trades else 0.0
    avg_top_profit = total_top_profit / total_trades if total_trades else 0.0

    print("Total Trades:", total_trades)
    print("Total Profit:", f"{total_profit:.2f}")
    print("Total Top Profit:", f"{total_top_profit:.2f}")
    print("Avg Profit:", f"{avg_profit:.2f}")
    print("Avg Top Profit:", f"{avg_top_profit:.2f}")
    print("Ticker List:")
    for i, tick_list in enumerate(ticker_history, start=1):
        print(f"Day {i}: {' '.join(tick_list)}")

    if daily_stats:
        plot_df = pd.DataFrame(daily_stats).set_index("date")
        plot_daily_results(plot_df)


if __name__ == "__main__":
    main()
