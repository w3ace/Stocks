from __future__ import annotations

import argparse

from tabulate import tabulate

from stocks.backtests.eldorado import backtest_open_range
from stocks.data.fetch import fetch_ticker
from stocks.utils.io import export_trades


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Eldorado open range backtest"
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--period", help="yfinance period, e.g. 1y")
    parser.add_argument("--profit-pct", type=float, default=1.0)
    parser.add_argument("--loss-pct", type=float, default=0.35)
    parser.add_argument("--export", help="Path to export trades as CSV")
    parser.add_argument(
        "--console-out", choices=["trades", "summary"], default="summary"
    )
    args = parser.parse_args()

    df = fetch_ticker(
        args.ticker, start=args.start, end=args.end, period=args.period
    )
    trades = backtest_open_range(
        df, profit_pct=args.profit_pct, loss_pct=args.loss_pct
    )
    export_trades(trades, args.export)

    if args.console_out == "trades":
        print(
            tabulate(
                trades, headers="keys", tablefmt="psql", showindex=False
            )
        )
    else:
        summary = {
            "trades": len(trades),
            "total_gain_pct": float(trades["gain_loss_pct"].sum())
            if not trades.empty
            else 0.0,
        }
        print(tabulate([summary], headers="keys", tablefmt="psql"))


if __name__ == "__main__":
    main()
