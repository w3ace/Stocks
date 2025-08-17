import argparse
from pathlib import Path

import pandas as pd

from stock_functions import period_to_start_end
from portfolio_utils import expand_ticker_args
from backtest_filters import fetch_daily_data, add_indicators


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with indicator statistics required by filters."""
    df = add_indicators(df)
    if df.empty:
        return df
    df = df.copy()
    df["TrendSlope"] = df["SMA20"] - df["SMA20_5dago"]
    df["GapPct"] = ((df["Open"] - df["PrevClose"]) / df["PrevClose"]).abs() * 100.0
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily indicator dataset")
    parser.add_argument("ticker", nargs="+", help="Ticker symbol or list of symbols")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--period", help="yfinance period string (e.g. 1y, 6mo)")
    group.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    tickers = expand_ticker_args(args.ticker)

    if args.start:
        start = pd.to_datetime(args.start)
        end = pd.to_datetime(args.end) if args.end else pd.Timestamp.now().normalize()
    else:
        start, end = period_to_start_end(args.period or "1y")
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

    enabled = {
        "avg_vol",
        "dollar_vol",
        "atr_pct",
        "nr7",
        "inside_2",
        "above_sma",
        "trend_slope",
        "pullback_pct",
        "gap",
    }
    req = 0
    if "avg_vol" in enabled:
        req = max(req, 20)
    if "dollar_vol" in enabled:
        req = max(req, 20)
    if "atr_pct" in enabled:
        req = max(req, 15)
    if "nr7" in enabled:
        req = max(req, 8)
    if "inside_2" in enabled:
        req = max(req, 3)
    if "above_sma" in enabled:
        req = max(req, 20)
    if "trend_slope" in enabled:
        req = max(req, 25)
    if "pullback_pct" in enabled:
        req = max(req, 20)
    if "gap" in enabled:
        req = max(req, 2)

    fetch_start = start - pd.Timedelta(days=req)
    fetch_end = end + pd.Timedelta(days=1)
    cache_tag = f"{start.date()}_{end.date()}"

    outdir = Path("datasets/indicators")
    outdir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        df = fetch_daily_data(ticker, fetch_start, fetch_end, cache_tag, "indicators")
        if df.empty:
            print(f"No data for {ticker}")
            continue
        df = compute_indicators(df)
        if df.empty:
            print(f"No indicators for {ticker}")
            continue
        df = df.iloc[req:]
        df = df[(df["Date"] >= start) & (df["Date"] <= end)]
        cols = [
            "Date",
            "VolSMA20",
            "DollarVol20",
            "ATRpct",
            "NR7",
            "Inside2",
            "SMA20",
            "SMA50",
            "SMA200",
            "SMA20_5dago",
            "TrendSlope",
            "PullbackPct20",
            "GapPct",
            "BodyPct",
            "UpperWickPct",
            "LowerWickPct",
        ]
        data = df[cols].copy()
        dest = outdir / f"{ticker}.csv"
        if dest.exists():
            existing = pd.read_csv(dest, parse_dates=["Date"])
            data = pd.concat([existing, data])
            data.drop_duplicates(subset="Date", keep="last", inplace=True)
            data.sort_values("Date", inplace=True)
        data.to_csv(dest, index=False)
        print(f"Saved {len(df)} rows to {dest}")


if __name__ == "__main__":
    main()
