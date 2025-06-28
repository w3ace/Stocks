import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


def run_backtest(arguments: list[str]) -> Path:
    """Run backtest.py with the given arguments and return the ticker csv path."""
    result = subprocess.run(["python", "backtest.py", *arguments], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("backtest.py failed")
    out_dir = Path("output")
    csvs = sorted(out_dir.glob("*_tickers.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError("Ticker summary csv not found")
    return csvs[0]


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

    while current <= end_date:
        lookback_start = current - timedelta(days=14)
        lookback_end = current - timedelta(days=1)

        # Run backtest on the lookback period to select tickers
        csv_path = run_backtest([
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
        result_csv = run_backtest([
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
        total_trades += result_df["total_trades"].sum()
        total_profit += result_df["total_profit"].sum()
        total_top_profit += result_df["total_top_profit"].sum()
        day_count += 1

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


if __name__ == "__main__":
    main()
