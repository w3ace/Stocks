import argparse
try:
    from tabulate import tabulate
except Exception:
    tabulate = None
from pathlib import Path
from typing import Iterable

import pandas as pd

from fetch_stock import fetch_stock


def expand_ticker_args(ticker_args: Iterable[str]) -> list[str]:
    """Expand any portfolio references in ``ticker_args``.

    Tokens beginning with ``+`` are treated as portfolio filenames under
    ``./portfolios``. The tickers listed inside are added to the result.
    """
    expanded: list[str] = []
    for token in ticker_args:
        if token.startswith("+"):
            name = token[1:]
            path = Path("portfolios") / name
            if path.exists():
                expanded.extend(path.read_text().split())
            else:
                print(f"Portfolio file not found: {path}")
        else:
            expanded.append(token)
    return expanded


def fetch_last_open_close(ticker: str) -> tuple[float, float] | None:
    """Return the last day's open and close for ``ticker``."""
    data = fetch_stock(ticker, period="7d", interval="1d")
    if data is None or data.empty:
        return None
    row = data.iloc[-1]
    try:
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        return open_price, close_price
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter tickers by last close vs open")
    parser.add_argument("ticker", nargs="+", help="Ticker symbol or +portfolio file")
    parser.add_argument(
        "--filter",
        choices=["LO", "OL", "ALL"],
        default="LO",
        help="LO: last close > open, OL: open > last close, ALL: no filter",
    )
    args = parser.parse_args()

    tickers = expand_ticker_args(args.ticker)
    rows: list[dict[str, float | str]] = []
    for t in tickers:
        result = fetch_last_open_close(t)
        if result is None:
            continue
        open_price, close_price = result
        if args.filter == "LO" and not close_price > open_price:
            continue
        if args.filter == "OL" and not open_price > close_price:
            continue
        pct_diff = (close_price - open_price) / open_price * 100 if open_price else 0
        rows.append({
            "ticker": t,
            "open": round(open_price, 2),
            "close": round(close_price, 2),
            "% diff": round(pct_diff, 2),
        })

    if rows:
        df = pd.DataFrame(rows)
        if tabulate:
            print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
        else:
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
