import argparse
import pandas as pd
import yfinance as yf
import sys


def fetch_market_cap(ticker: str) -> float:
    """Return the market capitalization for ``ticker`` using yfinance.

    On any error, 0 is returned.
    """
    try:
        info = yf.Ticker(ticker).info
        cap = info.get("marketCap")
        return float(cap) if cap is not None else 0.0
    except Exception as e:
        print(f"Failed to fetch market cap for {ticker}: {e}", file=sys.stderr)
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print tickers with market cap greater than the given value"
    )
    parser.add_argument(
        "min_cap",
        type=float,
        help="Minimum market capitalization (e.g. 1e9 for $1B)",
    )
    parser.add_argument(
        "--input",
        default="us_stocks.csv",
        help="CSV file containing a 'symbol' column with ticker symbols",
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Unable to read {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    tickers = df.get("symbol")
    if tickers is None:
        print(f"Column 'symbol' not found in {args.input}", file=sys.stderr)
        sys.exit(1)

    matches: list[str] = []
    for ticker in tickers.dropna().unique():
        cap = fetch_market_cap(ticker)
        if cap >= args.min_cap:
            matches.append(ticker)

    print(" ".join(matches))


if __name__ == "__main__":
    main()
