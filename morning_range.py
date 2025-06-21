import argparse
import os
import pandas as pd
from datetime import timedelta
import yfinance as yf


def fetch_intraday(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "1m",
    dataset_dir: str = "Datasets",
) -> pd.DataFrame:
    """Fetch intraday data for the ticker between start and end.

    First attempts to load local data from ``dataset_dir``. Only downloads
    missing data using :func:`yf.download` as a fallback.
    """

    frames = []
    first_letter = ticker[0].upper()
    all_dates = pd.date_range(start=start.normalize(), end=end.normalize())
    missing = False

    for day in all_dates:
        date_str = day.strftime("%Y%m%d")
        local_file = os.path.join(
            dataset_dir,
            "Ticker",
            "Daily",
            date_str,
            first_letter,
            ticker,
            f"{ticker}.bars",
        )
        if os.path.exists(local_file):
            df_day = pd.read_csv(local_file, index_col=0, parse_dates=True)
            if not isinstance(df_day.index, pd.DatetimeIndex):
                df_day.index = pd.to_datetime(df_day.index, errors="coerce")
            if df_day.index.tz is None:
                df_day.index = df_day.index.tz_localize("US/Eastern")
            df_day = df_day.tz_convert("UTC")
            frames.append(df_day)
        else:
            missing = True

    if missing:
        try:
            online = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
            )
        except Exception as exc:
            print(f"Failed download for {ticker}: {exc}")
            online = pd.DataFrame()

        if not online.empty:
            if isinstance(online.columns, pd.MultiIndex):
                online.columns = online.columns.get_level_values(0)
            if not isinstance(online.index, pd.DatetimeIndex):
                online.index = pd.to_datetime(online.index, errors="coerce")
            if online.index.tz is None:
                online.index = online.index.tz_localize("UTC")
            online = online.tz_convert("UTC")
            frames.append(online)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    df.sort_index(inplace=True)
    df = df.loc[(df.index >= start) & (df.index <= end + timedelta(days=1))]
    df = df[~df.index.duplicated(keep="first")]
    return df


def calculate_morning_range(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with daily high/low from 9:30 to 10:00 US/Eastern."""
    df = df.tz_convert("US/Eastern")
    morning = df.between_time("09:30", "10:00")
    if morning.empty:
        return pd.DataFrame()
    grouped = morning.groupby(morning.index.date)[["High", "Low"]].agg({"High": "max", "Low": "min"})
    grouped.index = pd.to_datetime(grouped.index)
    grouped.sort_index(inplace=True)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate high/low from 9:30 to 10:00am")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    start = pd.to_datetime(args.start).tz_localize("UTC")
    end = pd.to_datetime(args.end).tz_localize("UTC")

    for ticker in args.tickers:
        data = fetch_intraday(ticker, start, end, interval="1m")
        if data.empty:
            print(f"No 1m data for {ticker}. Trying 5m interval...")
            data = fetch_intraday(ticker, start, end, interval="5m")
        if data.empty:
            print(f"Unable to fetch intraday data for {ticker} in given range.")
            continue
        result = calculate_morning_range(data)
        if result.empty:
            print(f"No trading data between 9:30 and 10:00 for {ticker}.")
            continue
        print(f"\nMorning range for {ticker}:")
        print(result)


if __name__ == "__main__":
    main()
