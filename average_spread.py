import argparse
from datetime import datetime, timedelta
from typing import Iterable, List

import pandas as pd
from polygon import RESTClient


def fetch_quotes(
    client: RESTClient, ticker: str, days: int
) -> List:
    """Return all quote ticks for ``ticker`` over the last ``days`` days."""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    quotes = []
    for q in client.list_quotes(
        ticker,
        timestamp_gte=start,
        timestamp_lte=end,
        limit=50000,
    ):
        quotes.append(q)
    return quotes


def average_spread(quotes: Iterable) -> float | None:
    """Return the average bid/ask spread from ``quotes`` grouped in 5 minute buckets."""
    rows = [
        {
            "timestamp": pd.to_datetime(q.sip_timestamp, unit="ns", errors="coerce"),
            "spread": (q.ask_price or 0) - (q.bid_price or 0),
        }
        for q in quotes
        if q.ask_price is not None and q.bid_price is not None
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows).dropna()
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    spread = df["spread"].resample("5min").mean()
    return float(spread.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate average bid/ask spread from Polygon quotes"
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument(
        "--range",
        type=int,
        default=5,
        help="Number of days to analyze (max 60)",
    )
    args = parser.parse_args()

    days = min(max(args.range, 1), 60)

    client = RESTClient()

    for ticker in args.tickers:
        quotes = fetch_quotes(client, ticker, days)
        spread = average_spread(quotes)
        if spread is None:
            print(f"No data for {ticker}")
            continue
        print(f"{ticker}: average spread {spread:.4f}")


if __name__ == "__main__":
    main()
