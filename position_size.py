import argparse
import math
import pandas as pd

from fetch_stock import fetch_stock
from portfolio_utils import expand_ticker_args


def fetch_latest_price(ticker: str) -> float | None:
    """Return the most recent closing price for ``ticker``."""
    data = fetch_stock(ticker, period="1d")
    if data is None or data.empty:
        return None
    if "Datetime" in data.columns:
        data = data.sort_values("Datetime")
    return float(data.iloc[-1]["Close"])


def calculate_shares(price: float, budget: float) -> int:
    """Return number of shares whose cost is closest to ``budget``."""
    if price <= 0:
        return 0
    floor_shares = math.floor(budget / price)
    ceil_shares = math.ceil(budget / price)
    if abs(floor_shares * price - budget) <= abs(ceil_shares * price - budget):
        return floor_shares
    return ceil_shares


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate trade metrics for tickers")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols or +portfolio names")
    parser.add_argument("--stop-pct", type=float, default=0.2, help="Stop loss percent")
    parser.add_argument("--profit-pct", type=float, default=0.8, help="Profit target percent")
    parser.add_argument("--trade-size", type=float, default=1000.0, help="Dollar amount per trade")
    args = parser.parse_args()

    tickers = expand_ticker_args(args.tickers)

    rows: list[dict[str, float | int | str]] = []
    for ticker in tickers:
        price = fetch_latest_price(ticker)
        if price is None:
            print(f"Unable to fetch price for {ticker}")
            continue
        shares = calculate_shares(price, args.trade_size)
        stop_price = price * (1 - args.stop_pct / 100)
        profit_price = price * (1 + args.profit_pct / 100)
        rows.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "Stop": round(stop_price, 2),
            "Target": round(profit_price, 2),
            "Shares": shares,
        })

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    else:
        print("No data available")


if __name__ == "__main__":
    main()
