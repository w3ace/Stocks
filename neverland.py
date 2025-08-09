import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

try:
    from tabulate import tabulate
except Exception:  # ImportError or other
    tabulate = None

from fetch_stock import fetch_stock
from portfolio_utils import expand_ticker_args
from stock_functions import round_numeric_cols
from backtest_filters import fetch_daily_data, add_indicators, passes_filters


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


def closing_open_trades(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Return DataFrame of stats for the closing and opening ranges.

    ``minutes`` defines the length of both ranges. Each row contains the
    open, close, high and low for the buy (closing) range and the sell
    (opening) range along with gain metrics.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)
    dates = sorted(grouped.groups.keys())

    rows: list[dict[str, float | pd.Timestamp]] = []
    for i in range(len(dates) - 1):
        today = grouped.get_group(dates[i])
        tomorrow = grouped.get_group(dates[i + 1])

        close_start = (
            pd.Timestamp("16:00") - timedelta(minutes=minutes)
        ).strftime("%H:%M")
        closing = today.between_time(close_start, "16:00")
        if closing.empty:
            continue

        buy_open = closing.iloc[0]["Open"]
        buy_close = closing.iloc[-1]["Close"]
        buy_high = closing["High"].max()
        buy_low = closing["Low"].min()

        open_end = (
            pd.Timestamp("09:30") + timedelta(minutes=minutes)
        ).strftime("%H:%M")
        opening = tomorrow.between_time("09:30", open_end)
        if opening.empty:
            continue

        sell_open = opening.iloc[0]["Open"]
        sell_close = opening.iloc[-1]["Close"]
        sell_high = opening["High"].max()
        sell_low = opening["Low"].min()

        gain = sell_open - buy_close
        gain_pct = gain / buy_close * 100

        max_loss = sell_low - buy_high
        max_loss_pct = max_loss / buy_high * 100

        max_gain = sell_high - buy_low
        max_gain_pct = max_gain / buy_low * 100

        rows.append(
            {
                "buy_time": closing.index[-1],
                "sell_time": opening.index[0],
                "buy_open": float(buy_open),
                "buy_close": float(buy_close),
                "buy_low": float(buy_low),
                "buy_high": float(buy_high),
                "sell_open": float(sell_open),
                "sell_close": float(sell_close),
                "sell_low": float(sell_low),
                "sell_high": float(sell_high),
                "gain": float(gain),
                "gain_pct": float(gain_pct),
                "max_loss": float(max_loss),
                "max_loss_pct": float(max_loss_pct),
                "max_gain": float(max_gain),
                "max_gain_pct": float(max_gain_pct),
            }
        )

    return pd.DataFrame(rows)


def analyze_ticker(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp, minutes: int, args
) -> tuple[float, float, float, float, float, int, pd.DataFrame]:
    df = fetch_intraday(ticker, start, end + pd.Timedelta(days=1), interval="5m")
    trades = closing_open_trades(df, minutes)
    if trades.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, trades

    # Apply daily filters based on sell day (day1)
    cache_tag = f"{start.date()}_{end.date()}"
    fetch_start = start - pd.Timedelta(days=5)
    fetch_end = end + pd.Timedelta(days=1)
    daily_df = fetch_daily_data(ticker, fetch_start, fetch_end, cache_tag, "neverland")
    daily_df = add_indicators(daily_df)
    mask = []
    for _, row in trades.iterrows():
        sell_date = pd.to_datetime(row["sell_time"]).date()
        idx = daily_df.index[daily_df["Date"].dt.date == sell_date]
        if len(idx) and passes_filters(daily_df, int(idx[0]), args):
            mask.append(True)
        else:
            mask.append(False)
    trades = trades[mask]
    if trades.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, trades

    highest_gain = trades["gain_pct"].max()
    highest_loss = trades["gain_pct"].min()
    average_gain = trades["gain_pct"].mean()
    average_max_loss = trades["max_loss_pct"].mean()
    average_max_gain = trades["max_gain_pct"].mean()
    return (
        float(highest_gain),
        float(highest_loss),
        float(average_gain),
        float(average_max_loss),
        float(average_max_gain),
        len(trades),
        trades,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze closing and opening ranges. ``--range`` minutes defines "
            "both the buy and sell ranges."
        ),
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--range",
        type=int,
        default=30,
        help=(
            "Length in minutes of both the buy range (before close) and the "
            "sell range (after open)."
        ),
    )
    parser.add_argument(
        "--console-out",
        choices=["tickers", "trades"],
        help="Print per-ticker summary or trade details to console in an ASCII table",
    )
    parser.add_argument(
        "--max-out",
        type=int,
        default=25,
        help="Maximum results to display with --console-out tickers (default 25)",
    )
    # === Filtering flags ===
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=200.0)
    parser.add_argument("--min-avg-vol", dest="min_avg_vol", type=float, default=1_000_000)
    parser.add_argument("--min-dollar-vol", dest="min_dollar_vol", type=float, default=20_000_000)
    parser.add_argument("--min-atr-pct", type=float, default=1.0)
    parser.add_argument("--max-atr-pct", type=float, default=8.0)
    parser.add_argument("--above-sma", type=int, choices=[20, 50, 200], default=20)
    parser.add_argument("--trend-slope", type=float, default=0.0)
    parser.add_argument("--nr7", action="store_true")
    parser.add_argument("--inside-2", dest="inside_2", action="store_true")
    parser.add_argument("--min-gap-pct", type=float, default=0.4)
    parser.add_argument("--body-pct-min", dest="body_pct_min", type=float, default=60.0)
    parser.add_argument("--upper-wick-max", dest="upper_wick_max", type=float, default=30.0)
    parser.add_argument("--lower-wick-max", dest="lower_wick_max", type=float, default=40.0)
    parser.add_argument("--pullback-pct-max", dest="pullback_pct_max", type=float, default=6.0)
    args = parser.parse_args()

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else start

    tickers = expand_ticker_args(args.tickers)

    rows: list[dict[str, float | str | int]] = []
    trades_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for ticker in tickers:
        hi, lo, avg, avg_max_loss, avg_max_gain, count, trades = analyze_ticker(
            ticker, start, end, args.range, args
        )
        if count == 0:
            print(f"{ticker}: no data in range")
            continue
        rows.append(
            {
                "ticker": ticker,
                "highest_gain": hi,
                "highest_loss": lo,
                "average_gain": avg,
                "average_max_loss": avg_max_loss,
                "average_max_gain": avg_max_gain,
                "count": count,
            }
        )
        if not trades.empty:
            tdf = trades.copy()
            tdf["ticker"] = ticker
            trades_rows.extend(tdf.to_dict(orient="records"))

    if args.console_out == "tickers" and rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(by="average_gain", ascending=False)
        df = round_numeric_cols(df)
        df = df.head(args.max_out)
        if tabulate:
            print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
        else:
            print(df.to_string(index=False))

        dest_dir = Path("tickers") / "neverland"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{start.strftime('%Y-%m-%d')}-{end.strftime('%Y-%m-%d')}.csv"
        df.to_csv(dest_file, index=False)
        print(f"Ticker summary saved to {dest_file}")

    if args.console_out == "trades" and trades_rows:
        trades_df = pd.DataFrame(trades_rows)
        trades_df = trades_df.sort_values(by="buy_time")
        for col in ["buy_time", "sell_time"]:
            trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime("%Y-%m-%d %H:%M")
        trades_df = round_numeric_cols(trades_df)
        if tabulate:
            print(tabulate(trades_df, headers="keys", tablefmt="grid", showindex=False))
        else:
            print(trades_df.to_string(index=False))

        dest_dir = Path("trades") / "neverland"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{start.strftime('%Y-%m-%d')}-{end.strftime('%Y-%m-%d')}.csv"
        trades_df.to_csv(dest_file, index=False)
        print(f"Trades saved to {dest_file}")

    total_trades = len(trades_rows)
    total_tickers = len(rows)
    avg_gain = (
        sum(r["gain_pct"] for r in trades_rows) / total_trades if total_trades else 0.0
    )
    print(
        f"Totals - trades: {total_trades}, tickers: {total_tickers}, avg gain: {avg_gain:.2f}%"
    )


if __name__ == "__main__":
    main()
