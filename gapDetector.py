import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
from portfolio_utils import expand_ticker_args
from backtest_filters import fetch_daily_data as fetch_cached_daily_data
from stock_functions import period_to_start_end, round_numeric_cols

REGULAR_MARKET_OPEN = pd.Timestamp("09:30").time()
REGULAR_MARKET_CLOSE = pd.Timestamp("16:00").time()

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - optional dependency
    tabulate = None


def parse_date_range(
    period: str | None, start: str | None, end: str | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
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


def fetch_daily_data(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp, cache_tag: str
) -> pd.DataFrame:
    """Fetch daily bars for *ticker* with one extra prior day for gap detection."""
    fetch_start = start - pd.Timedelta(days=10)
    fetch_end = end + pd.Timedelta(days=1)
    df = fetch_cached_daily_data(
        ticker, fetch_start, fetch_end, cache_tag, "gapDetector"
    )
    if df.empty:
        return df
    df = flatten_yfinance_columns(df)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def is_regular_market_time(timestamp: pd.Timestamp) -> bool:
    """Return True when *timestamp* falls inside regular US market hours."""
    ts = pd.Timestamp(timestamp)
    # yfinance returns exchange-local timestamps for this history call, so use the
    # timestamp's displayed wall-clock time instead of requiring local tzdata.
    market_time = ts.time()
    return REGULAR_MARKET_OPEN <= market_time <= REGULAR_MARKET_CLOSE


def fetch_current_extended_gap(ticker: str) -> dict[str, float | str] | None:
    """Return the current extended-hours gap for *ticker*, if available."""
    history = yf.Ticker(ticker).history(
        period="5d", interval="1m", prepost=True, auto_adjust=False
    )
    if history.empty or "Close" not in history.columns:
        return None

    history = flatten_yfinance_columns(history).dropna(subset=["Close"]).copy()
    if history.empty:
        return None

    if not isinstance(history.index, pd.DatetimeIndex):
        history.index = pd.to_datetime(history.index)

    latest_row = history.iloc[-1]
    latest_timestamp = pd.Timestamp(history.index[-1])
    if is_regular_market_time(latest_timestamp):
        return None

    regular_rows = history[
        history.index.map(is_regular_market_time) & (history.index < latest_timestamp)
    ]
    if regular_rows.empty:
        return None

    previous_close = float(regular_rows.iloc[-1]["Close"])
    latest_price = float(latest_row["Close"])
    if previous_close == 0:
        return None

    current_gap_pct = (latest_price - previous_close) / previous_close * 100
    direction = (
        "up" if current_gap_pct > 0 else "down" if current_gap_pct < 0 else "flat"
    )
    return {
        "ticker": ticker,
        "current_gap_pct": current_gap_pct,
        "current_extended_price": latest_price,
        "previous_regular_close": previous_close,
        "current_gap_direction": direction,
        "current_extended_timestamp": latest_timestamp.isoformat(),
    }


def current_gap_tickers(
    tickers: list[str], tolerance: float, direction: str
) -> dict[str, dict[str, float | str]]:
    """Return tickers currently gapping in the requested extended-hours direction."""
    min_gap = abs(tolerance)
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")

    current_gaps = {}
    for ticker in tickers:
        gap = fetch_current_extended_gap(ticker)
        if gap is None:
            continue
        current_gap_pct = float(gap["current_gap_pct"])
        meets_tolerance = (
            current_gap_pct >= min_gap
            if direction == "up"
            else current_gap_pct <= -min_gap
        )
        if min_gap == 0:
            meets_tolerance = (
                current_gap_pct > 0 if direction == "up" else current_gap_pct < 0
            )
        if meets_tolerance:
            current_gaps[ticker] = gap
    return current_gaps


def analyze_gaps(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tolerance: float,
    success: float,
    cache_tag: str,
) -> dict[str, float | int | str]:
    """Summarize significant previous-close-to-open gaps for a ticker."""
    tolerance = abs(tolerance)
    df = fetch_daily_data(ticker, start, end, cache_tag)
    if df.empty:
        return {
            "ticker": ticker,
            "days_analyzed": 0,
            "gap_up_days": 0,
            "gap_down_days": 0,
            "gap_up_days_pct": 0.0,
            "gap_down_days_pct": 0.0,
            "avg_after_gap_up_pct": 0.0,
            "avg_after_gap_down_pct": 0.0,
            "avg_max_up_pct": 0.0,
            "avg_max_down_pct": 0.0,
            "success_up_pct": 0.0,
            "success_down_pct": 0.0,
        }

    df = df.sort_values("Date").copy()
    df["prev_close"] = df["Close"].shift(1)
    df["gap_pct"] = (df["Open"] - df["prev_close"]) / df["prev_close"] * 100
    df["after_gap_pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100
    df["max_up_pct"] = (df["High"] - df["Open"]) / df["Open"] * 100
    df["max_down_pct"] = (df["Low"] - df["Open"]) / df["Open"] * 100

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
    days_analyzed = len(in_range)
    gap_up_count = len(gap_up)
    gap_down_count = len(gap_down)
    success = abs(success)
    successful_gap_up_count = (
        int(
            (
                (gap_up["after_gap_pct"] >= success)
                | (gap_up["max_up_pct"] >= success * 2)
            ).sum()
        )
        if gap_up_count
        else 0
    )
    successful_gap_down_count = (
        int(
            (
                (gap_down["after_gap_pct"] <= -success)
                | (gap_down["max_down_pct"] <= -(success * 2))
            ).sum()
        )
        if gap_down_count
        else 0
    )
    return {
        "ticker": ticker,
        "days_analyzed": int(days_analyzed),
        "gap_up_days": int(gap_up_count),
        "gap_down_days": int(gap_down_count),
        "gap_up_days_pct": (
            (gap_up_count / days_analyzed * 100) if days_analyzed else 0.0
        ),
        "gap_down_days_pct": (
            (gap_down_count / days_analyzed * 100) if days_analyzed else 0.0
        ),
        "avg_after_gap_up_pct": (
            float(gap_up["after_gap_pct"].mean()) if not gap_up.empty else 0.0
        ),
        "avg_after_gap_down_pct": (
            float(gap_down["after_gap_pct"].mean()) if not gap_down.empty else 0.0
        ),
        "avg_max_up_pct": (
            float(gap_up["max_up_pct"].mean()) if not gap_up.empty else 0.0
        ),
        "avg_max_down_pct": (
            float(gap_down["max_down_pct"].mean()) if not gap_down.empty else 0.0
        ),
        "success_up_pct": (
            (successful_gap_up_count / gap_up_count * 100) if gap_up_count else 0.0
        ),
        "success_down_pct": (
            (successful_gap_down_count / gap_down_count * 100)
            if gap_down_count
            else 0.0
        ),
    }


def unique_output_path(filename: str | None) -> Path:
    """Return a non-existing path in the output directory, adding a digit if needed."""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(filename).name if filename else "gap_summary.csv"
    output_path = output_dir / base_name
    if not output_path.exists():
        return output_path

    stem = output_path.stem
    suffix = output_path.suffix
    counter = 1
    while True:
        candidate = output_dir / f"{stem}{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def filter_report_columns(summary: pd.DataFrame, report: str) -> pd.DataFrame:
    """Limit gap summary columns to the requested report direction."""
    shared_columns = ["ticker", "days_analyzed"]
    up_columns = [
        "gap_up_days",
        "gap_up_days_pct",
        "avg_after_gap_up_pct",
        "avg_max_up_pct",
        "success_up_pct",
    ]
    down_columns = [
        "gap_down_days",
        "gap_down_days_pct",
        "avg_after_gap_down_pct",
        "avg_max_down_pct",
        "success_down_pct",
    ]
    current_gap_columns = [
        "current_gap_pct",
        "current_extended_price",
        "previous_regular_close",
        "current_gap_direction",
        "current_extended_timestamp",
    ]

    if report == "up":
        columns = shared_columns + up_columns
    elif report == "down":
        columns = shared_columns + down_columns
    else:
        raise ValueError("report must be 'up' or 'down'")

    columns += [column for column in current_gap_columns if column in summary.columns]
    return summary.reindex(columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect significant gap-up and gap-down days."
    )
    parser.add_argument(
        "ticker", nargs="+", help="Ticker symbol(s), or portfolio names prefixed with +"
    )
    parser.add_argument(
        "--period",
        help="yfinance period string (e.g. 1y, 6mo) (overrides start/end if provided)",
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Minimum absolute previous-close-to-open gap percentage required to count (default 0)",
    )
    parser.add_argument(
        "--success",
        type=float,
        default=0.0,
        help=(
            "Open-to-close percentage threshold for success rates. Gap-up days count as successful "
            "when after-gap pct is >= this value; gap-down days count when <= the negative value "
            "(default 0)."
        ),
    )
    parser.add_argument(
        "--report",
        choices=("up", "down", "gap_up", "gap_down"),
        required=True,
        help=(
            "Gap direction data to report. Use gap_up or gap_down to first limit "
            "analysis to tickers currently gapping in that direction in premarket "
            "or after-hours data."
        ),
    )
    parser.add_argument(
        "--csv-out",
        help="Optional CSV filename for the ticker summary output; files are always saved under output/",
    )
    args = parser.parse_args()

    tickers = expand_ticker_args(args.ticker)
    start, end = parse_date_range(args.period, args.start, args.end)
    cache_tag = f"{start.date()}_{end.date()}"
    current_gaps = {}
    report = args.report
    if report in {"gap_up", "gap_down"}:
        current_gap_direction = "up" if report == "gap_up" else "down"
        current_gaps = current_gap_tickers(
            tickers, args.tolerance, current_gap_direction
        )
        tickers = list(current_gaps)
        report = current_gap_direction

    rows = [
        analyze_gaps(ticker, start, end, args.tolerance, args.success, cache_tag)
        for ticker in tickers
    ]
    summary = round_numeric_cols(pd.DataFrame(rows))
    if current_gaps and not summary.empty:
        current_gap_summary = pd.DataFrame(current_gaps.values())
        summary = summary.merge(current_gap_summary, on="ticker", how="left")
    summary = filter_report_columns(summary, report)
    success_column = "success_up_pct" if report == "up" else "success_down_pct"
    if success_column in summary.columns:
        summary = summary[summary[success_column] >= 50]

    if tabulate:
        print(tabulate(summary, headers="keys", tablefmt="grid", showindex=False))
    else:
        print(summary.to_string(index=False))

    output_path = unique_output_path(args.csv_out)
    summary.to_csv(output_path, index=False)
    print(f"Gap summary saved to {output_path}")


if __name__ == "__main__":
    main()
