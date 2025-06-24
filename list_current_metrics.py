import argparse
import math
import pandas as pd
from fetch_stock import fetch_stock


def fetch_latest_price(ticker: str) -> float | None:
    """Return the most recent closing price for ``ticker`` using :func:`fetch_stock`."""
    data = fetch_stock(ticker, period="1d")
    if data is None or data.empty:
        return None
    # Ensure we have a datetime index
    if "Datetime" in data.columns:
        data = data.sort_values("Datetime")
    return float(data.iloc[-1]["Close"])


def calculate_shares(price: float, budget: float = 1000.0) -> int:
    """Return number of shares closest to ``budget`` for the given ``price``."""
    if price <= 0:
        return 0
    floor_shares = math.floor(budget / price)
    ceil_shares = math.ceil(budget / price)
    # Choose shares whose cost is closest to budget
    if abs(floor_shares * price - budget) <= abs(ceil_shares * price - budget):
        return floor_shares
    return ceil_shares


def main() -> None:
    parser = argparse.ArgumentParser(description="List current trade metrics for tickers")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--loss-pct", type=float, default=0.2, help="Stop loss percent")
    parser.add_argument("--profit-pct", type=float, default=0.8, help="Profit target percent")
    args = parser.parse_args()

    rows: list[dict[str, str | float | int]] = []
    for ticker in args.tickers:
        price = fetch_latest_price(ticker)
        if price is None:
            print(f"Unable to fetch price for {ticker}")
            continue
        shares = calculate_shares(price)
        stop_price = price * (1 - args.loss_pct / 100)
        profit_price = price * (1 + args.profit_pct / 100)
        rows.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "Shares": shares,
            "Stop": round(stop_price, 2),
            "Target": round(profit_price, 2),
        })

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    else:
        print("No data available")


if __name__ == "__main__":
    main()
