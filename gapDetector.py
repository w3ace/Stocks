import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from portfolio_utils import expand_ticker_args
from stock_functions import period_to_start_end, round_numeric_cols

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - optional dependency
    tabulate = None


def parse_date_range(period: str | None, start: str | None, end: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive start/end timestamps from CLI date parameters."""
    if period:
        start_dt, end_dt = period_to_start_end(period)
        return pd.to_datetime(start_dt), pd.to_datetime(end_dt)
    if start:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) if end else pd.Timestamp.now().normalize()
        return start_dt, end_dt
    start_dt, end_dt = period_to_start_end("1y")
    return pd.to_datetime(start_dt), pd.to_datetime(end_dt)


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output so OHLC columns are simple strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_daily_data(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download daily bars for *ticker* with one extra prior day for gap detection."""
    download_start = (start - timedelta(days=10)).strftime("%Y-%m-%d")
    download_end = (end + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        ticker,
        start=download_start,
        end=download_end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        return df
    df = flatten_yfinance_columns(df).reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def analyze_gaps(ticker: str, start: pd.Timestamp, end: pd.Timestamp, tolerance: float) -> dict[str, float | int | str]:
    """Summarize significant previous-close-to-open gaps for a ticker."""
    tolerance = abs(tolerance)
    df = fetch_daily_data(ticker, start, end)
    if df.empty:
        return {
            "ticker": ticker,
            "days_analyzed": 0,
            "gap_up_days": 0,
            "gap_down_days": 0,
            "gap_days": 0,
            "avg_gap_pct": 0.0,
            "avg_after_gap_pct": 0.0,
            "avg_after_gap_up_pct": 0.0,
            "avg_after_gap_down_pct": 0.0,
        }

    df = df.sort_values("Date").copy()
    df["prev_close"] = df["Close"].shift(1)
    df["gap_pct"] = (df["Open"] - df["prev_close"]) / df["prev_close"] * 100
    df["after_gap_pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100

    in_range = df[
        (df["Date"].dt.date >= start.date())
        & (df["Date"].dt.date <= end.date())
        & df["prev_close"].notna()
    ].copy()
    if tolerance == 0:
        gap_up = in_range[in_range["gap_pct"] > 0]
        gap_down = in_range[in_range["gap_pct"] < 0]
    else:
        gap_up = in_range[in_range["gap_pct"] >= tolerance]
        gap_down = in_range[in_range["gap_pct"] <= -tolerance]
    gap_days = pd.concat([gap_up, gap_down]).sort_values("Date")

    return {
        "ticker": ticker,
        "days_analyzed": int(len(in_range)),
        "gap_up_days": int(len(gap_up)),
        "gap_down_days": int(len(gap_down)),
        "gap_days": int(len(gap_days)),
        "avg_gap_pct": float(gap_days["gap_pct"].mean()) if not gap_days.empty else 0.0,
        "avg_after_gap_pct": float(gap_days["after_gap_pct"].mean()) if not gap_days.empty else 0.0,
        "avg_after_gap_up_pct": float(gap_up["after_gap_pct"].mean()) if not gap_up.empty else 0.0,
        "avg_after_gap_down_pct": float(gap_down["after_gap_pct"].mean()) if not gap_down.empty else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect significant gap-up and gap-down days.")
    parser.add_argument("ticker", nargs="+", help="Ticker symbol(s), or portfolio names prefixed with +")
    parser.add_argument("--period", help="yfinance period string (e.g. 1y, 6mo) (overrides start/end if provided)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Minimum absolute previous-close-to-open gap percentage required to count (default 0)",
    )
    parser.add_argument(
        "--csv-out",
        help="Optional CSV path for the ticker summary output",
    )
    args = parser.parse_args()

    tickers = expand_ticker_args(args.ticker)
    start, end = parse_date_range(args.period, args.start, args.end)
    rows = [analyze_gaps(ticker, start, end, args.tolerance) for ticker in tickers]
    summary = round_numeric_cols(pd.DataFrame(rows))

    if tabulate:
        print(tabulate(summary, headers="keys", tablefmt="grid", showindex=False))
    else:
        print(summary.to_string(index=False))

    if args.csv_out:
        output_path = Path(args.csv_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"Gap summary saved to {output_path}")


if __name__ == "__main__":
    main()
