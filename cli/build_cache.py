from __future__ import annotations

import argparse

from stocks.data.fetch import fetch_ticker


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fetch data for tickers")
    parser.add_argument("--ticker", action="append", required=True)
    parser.add_argument("--period", default="1y")
    args = parser.parse_args()
    for ticker in args.ticker:
        fetch_ticker(ticker, period=args.period, force=True)


if __name__ == "__main__":
    main()
