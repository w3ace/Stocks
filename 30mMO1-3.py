import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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
        if current.weekday() >= 5:  # Skip Saturday and Sunday
            current += timedelta(days=1)
            continue
        lookback_start = current - timedelta(days=14)
        lookback_end = current - timedelta(days=1)

        # Run backtest on the lookback period to select tickers
        csv_path, _ = run_backtest([
            "--end", lookback_end.strftime("%Y-%m-%d"),
            "--start", lookback_start.strftime("%Y-%m-%d"),
            "--loss-pct", "0.35",
            "--profit-pct", "1.05",
            "--range", "30",
            "--filter", "MO",
            "--min-profit", "-1",
            "+30mMO-L"
        ])

        df = pd.read_csv(csv_path).sort_values(by="total_profit", ascending=False)
        tickers = df["ticker"].head(20).tolist()
        if not tickers:
            current += timedelta(days=1)
            continue

        ticker_history.append(tickers)

        # Run backtest for the current day using selected tickers
        result_csv, trades_csv = run_backtest([
            "--end", current.strftime("%Y-%m-%d"),
            "--start", current.strftime("%Y-%m-%d"),
            "--loss-pct", "0.35",
            "--profit-pct", "1.05",
            "--range", "30",
            "--filter", "MO",
            "--min-profit", "-1",
            *tickers,
        ])

        result_df = pd.read_csv(result_csv)
        trades_df = pd.read_csv(trades_csv)

        total_trades += result_df["total_trades"].sum()
        total_profit += result_df["total_profit"].sum()
        total_top_profit += result_df["total_top_profit"].sum()
        day_count += 1

        avg_profit_day = trades_df["profit"].mean() if "profit" in trades_df.columns else 0.0
        avg_top_profit_day = (
            trades_df["top_profit"].mean() if "top_profit" in trades_df.columns else 0.0
        )
        counts = trades_df["result"].value_counts() if "result" in trades_df.columns else pd.Series(dtype=int)
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

    avg_profit = total_profit / day_count if day_count else 0.0
    avg_top_profit = total_top_profit / day_count if day_count else 0.0

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
