import argparse
from datetime import timedelta

import pandas as pd

from fetch_stock import fetch_stock
from portfolio_utils import expand_ticker_args


def fetch_intraday(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str = "5m") -> pd.DataFrame:
    """Fetch intraday data for ``ticker`` using :func:`fetch_stock`."""
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    start_est = start.tz_localize("US/Eastern") if start.tzinfo is None else start.tz_convert("US/Eastern")
    start_est = start_est.normalize() + pd.Timedelta(hours=9, minutes=30)
    start = start_est.tz_convert("UTC")

    end_est = end.tz_localize("US/Eastern") if end.tzinfo is None else end.tz_convert("US/Eastern")
    if end_est.time() == pd.Timestamp("00:00", tz="US/Eastern").time():
        end_est = end_est.normalize() + pd.Timedelta(hours=16)
    end = end_est.tz_convert("UTC")

    data = fetch_stock(ticker, start_date=start, end_date=end, interval=interval)
    if data is None or data.empty:
        return pd.DataFrame()

    if "Datetime" in data.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(data["Datetime"], errors="coerce"))
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(data.index, errors="coerce"))

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    data.index = idx
    if "Datetime" in data.columns:
        data = data.drop(columns=["Datetime"])

    data.sort_index(inplace=True)
    data = data.loc[(data.index >= start) & (data.index <= end + timedelta(days=1))]
    return data


def closing_open_gains(df: pd.DataFrame, minutes: int) -> list[float]:
    """Return list of percent gains from closing range high to next day opening range low."""
    if df.empty:
        return []

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)
    dates = sorted(grouped.groups.keys())

    gains: list[float] = []
    for i in range(len(dates) - 1):
        today = grouped.get_group(dates[i])
        tomorrow = grouped.get_group(dates[i + 1])

        close_start = (pd.Timestamp("16:00") - timedelta(minutes=minutes)).strftime("%H:%M")
        closing = today.between_time(close_start, "16:00")
        if closing.empty:
            continue
        closing_high = closing["High"].max()

        open_end = (pd.Timestamp("09:30") + timedelta(minutes=minutes)).strftime("%H:%M")
        opening = tomorrow.between_time("09:30", open_end)
        if opening.empty:
            continue
        opening_low = opening["Low"].min()

        gain_pct = (opening_low - closing_high) / closing_high * 100
        gains.append(float(gain_pct))

    return gains


def analyze_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp, minutes: int) -> tuple[float, float, float, int]:
    df = fetch_intraday(ticker, start, end + pd.Timedelta(days=1), interval="5m")
    gains = closing_open_gains(df, minutes)
    if not gains:
        return 0.0, 0.0, 0.0, 0
    highest_gain = max(gains)
    highest_loss = min(gains)
    average_gain = sum(gains) / len(gains)
    return highest_gain, highest_loss, average_gain, len(gains)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate overnight gains from closing range high to next day opening range low",
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--range",
        type=int,
        default=30,
        help="Closing/opening range in minutes (default 30)",
    )
    args = parser.parse_args()

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else start

    tickers = expand_ticker_args(args.tickers)

    for ticker in tickers:
        hi, lo, avg, count = analyze_ticker(ticker, start, end, args.range)
        if count == 0:
            print(f"{ticker}: no data in range")
            continue
        print(
            f"{ticker}: highest gain {hi:.2f}% | highest loss {lo:.2f}% | average {avg:.2f}% (n={count})"
        )


if __name__ == "__main__":
    main()
