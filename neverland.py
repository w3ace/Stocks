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
    """Return DataFrame of trades from closing range high to next day opening range low."""
    if df.empty:
        return pd.DataFrame()

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)
    dates = sorted(grouped.groups.keys())

    rows: list[dict[str, float | pd.Timestamp]] = []
    for i in range(len(dates) - 1):
        today = grouped.get_group(dates[i])
        tomorrow = grouped.get_group(dates[i + 1])

        close_start = (pd.Timestamp("16:00") - timedelta(minutes=minutes)).strftime("%H:%M")
        closing = today.between_time(close_start, "16:00")
        if closing.empty:
            continue
        buy_idx = closing["Close"].idxmax()
        buy_price = closing.loc[buy_idx, "Close"]

        open_end = (pd.Timestamp("09:30") + timedelta(minutes=minutes)).strftime("%H:%M")
        opening = tomorrow.between_time("09:30", open_end)
        if opening.empty:
            continue
        sell_idx = opening["Open"].idxmin()
        sell_price = opening.loc[sell_idx, "Open"]

        gain = sell_price - buy_price
        gain_pct = gain / buy_price * 100

        rows.append(
            {
                "buy_time": buy_idx,
                "sell_time": sell_idx,
                "buy_price": float(buy_price),
                "sell_price": float(sell_price),
                "gain": float(gain),
                "gain_pct": float(gain_pct),
            }
        )

    return pd.DataFrame(rows)


def analyze_ticker(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp, minutes: int
) -> tuple[float, float, float, int, pd.DataFrame]:
    df = fetch_intraday(ticker, start, end + pd.Timedelta(days=1), interval="5m")
    trades = closing_open_trades(df, minutes)
    if trades.empty:
        return 0.0, 0.0, 0.0, 0, trades
    highest_gain = trades["gain_pct"].max()
    highest_loss = trades["gain_pct"].min()
    average_gain = trades["gain_pct"].mean()
    return float(highest_gain), float(highest_loss), float(average_gain), len(trades), trades


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
    args = parser.parse_args()

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else start

    tickers = expand_ticker_args(args.tickers)

    rows: list[dict[str, float | str | int]] = []
    trades_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for ticker in tickers:
        hi, lo, avg, count, trades = analyze_ticker(ticker, start, end, args.range)
        if count == 0:
            print(f"{ticker}: no data in range")
            continue
        rows.append(
            {
                "ticker": ticker,
                "highest_gain": hi,
                "highest_loss": lo,
                "average_gain": avg,
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
