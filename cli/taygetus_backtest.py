from __future__ import annotations

import argparse

from tabulate import tabulate

from stocks.backtests.taygetus import backtest_pattern
from stocks.data.fetch import fetch_ticker
from stocks.utils.io import export_trades


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Taygetus pattern backtest"
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument(
        "--pattern", default="3E", help="Pattern filter, e.g. 3E"
        )
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--period",
        help="yfinance period, e.g. 1y (overrides start/end if both provided)",
    )
    parser.add_argument("--export", help="Path to export trades as CSV")
    parser.add_argument(
        "--console-out",
        choices=["trades", "summary"],
        default="summary",
    )
    args = parser.parse_args()

    # yfinance ignores ``period`` when ``start``/``end`` are supplied.  For the
    # CLI we explicitly prefer ``period`` if both are given.
    if args.period:
        df = fetch_ticker(args.ticker, period=args.period)
    else:
        df = fetch_ticker(args.ticker, start=args.start, end=args.end)
    trades = backtest_pattern(df, args.pattern)
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
            "total_gain_pct": float(
                trades["gain_loss_pct"].sum()
            ) if not trades.empty else 0.0,
        }
        print(tabulate([summary], headers="keys", tablefmt="psql"))


if __name__ == "__main__":
    main()
