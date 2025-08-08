import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
from stock_functions import period_to_start_end, round_numeric_cols
from portfolio_utils import expand_ticker_args

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - optional dependency
    tabulate = None


CACHE_DIR = Path(__file__).resolve().parent / "yfinance_cache" / "taygetus"


def fetch_daily_data(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_tag: str,
) -> pd.DataFrame:
    """Download daily price data for *ticker* between *start* and *end*.

    The raw *yfinance* response is cached under ``CACHE_DIR/cache_tag`` grouped by
    the first letter of ``ticker``. Caching is disabled if the requested range
    includes the current trading day and it is before 4:30pm US/Eastern.
    """

    now_est = pd.Timestamp.now(tz="US/Eastern")
    four_thirty = pd.Timestamp("16:30", tz="US/Eastern").time()
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()

    cache_enabled = not (
        start_date <= now_est.date() < end_date and now_est.time() < four_thirty
    )

    cache_file = CACHE_DIR / cache_tag / ticker[0].upper() / ticker

 #   print("Looking for ",cache_file, cache_enabled)

    if cache_enabled and cache_file.exists():
        try:
            data = pd.read_pickle(cache_file)
        except Exception:
            data = pd.DataFrame()
    else:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if cache_enabled and not data.empty:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                data.to_pickle(cache_file)
            except Exception:
                pass

    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Open", "High", "Low", "Close"]]
    data.reset_index(inplace=True)
    return data


def fetch_current_price(ticker: str) -> float | None:
    """Return the latest price for ``ticker`` using 1 minute data."""
    data = yf.download(
        ticker,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data["Close"].iloc[-1].item()


def backtest_pattern(
    df: pd.DataFrame, filter_value: str
) -> list[dict[str, float | pd.Timestamp]]:
    """Return detailed trades meeting the selected Taygetus pattern.

    Parameters
    ----------
    df : pd.DataFrame
        Daily price data.
    filter_value : str
        One of ``"3E"``, ``"4E"``, ``"3D"`` or ``"4D"``. If the first
        character is a digit, it determines the number of pattern days. ``day1``
        refers to the most recent day (exit day) and numbering counts backwards.

    Notes
    -----
    Trades are entered at ``day2`` close and evaluated at ``day1`` open, close,
    high and low. Gain/loss percentages are calculated for each of these
    potential exits.
    """

    pattern_length = (
        int(filter_value[0]) + 1 if filter_value and filter_value[0].isdigit() else 4
    )

    trades: list[dict[str, float | pd.Timestamp]] = []
    for i in range(pattern_length - 1, len(df)):
        days = {f"day{j + 1}": df.iloc[i - j] for j in range(pattern_length)}

        entry_price = None
        if filter_value == "3E":
            if (
                days["day4"]["Close"] > days["day4"]["Open"]
                and days["day3"]["Close"] > days["day3"]["Open"]
                and days["day2"]["Open"] > days["day3"]["Close"]
                and days["day2"]["Close"] < days["day3"]["Open"]
            ):
                entry_price = days["day2"]["Close"]
        elif filter_value == "4E":
            if (
                days["day5"]["Close"] > days["day5"]["Open"]
                and days["day4"]["Close"] > days["day4"]["Open"]
                and days["day3"]["Close"] > days["day3"]["Open"]
                and days["day2"]["Open"] > days["day3"]["Close"]
                and days["day2"]["Close"] < days["day3"]["Open"]
            ):
                entry_price = days["day2"]["Close"]
        elif filter_value == "3D":
            if days["day4"]["Close"] > days["day3"]["Close"] > days["day2"]["Close"]:
                entry_price = days["day2"]["Close"]
        elif filter_value == "4D":
            if (
                days["day5"]["Close"]
                > days["day4"]["Close"]
                > days["day3"]["Close"]
                > days["day2"]["Close"]
            ):
                entry_price = days["day2"]["Close"]
        else:
            continue

        if entry_price is not None:
            exit_open = days["day1"]["Open"]
            exit_close = days["day1"]["Close"]
            exit_high = days["day1"]["High"]
            exit_low = days["day1"]["Low"]

            gain_open = exit_open - entry_price
            gain_close = exit_close - entry_price
            gain_high = exit_high - entry_price
            gain_low = exit_low - entry_price

            trades.append(
                {
                    "entry_day": days["day2"]["Date"].date(),
                    "exit_day": days["day1"]["Date"].date(),
                    "entry_price": entry_price,
                    "exit_open": exit_open,
                    "exit_close": exit_close,
                    "exit_high": exit_high,
                    "exit_low": exit_low,
                    "gain_loss_open": gain_open,
                    "gain_loss_close": gain_close,
                    "gain_loss_high": gain_high,
                    "gain_loss_low": gain_low,
                    "gain_loss_open_pct": gain_open / entry_price * 100,
                    "gain_loss_close_pct": gain_close / entry_price * 100,
                    "gain_loss_high_pct": gain_high / entry_price * 100,
                    "gain_loss_low_pct": gain_low / entry_price * 100,
                }
            )
    return trades


def main() -> None:
    parser = argparse.ArgumentParser(description="Taygetus pattern backtest")
    parser.add_argument('ticker', nargs='+', help='Ticker symbol or list of symbols')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--period', help='yfinance period string (e.g. 1y, 6mo)')
    group.add_argument('--start', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', help='Last buy date YYYY-MM-DD')
    parser.add_argument(
        '--console-out',
        choices=['tickers', 'trades'],
        help='Print per-ticker or per-trade summary to console in an ASCII table',
    )
    parser.add_argument(
        '--filter',
        choices=['3E', '4E', '3D', '4D'],
        default='3E',
        help='Pattern filter: 3E current Taygetus, 3D descending closes',
    )
    parser.add_argument(
        '--max-out',
        type=int,
        default=40,
        help='Maximum tickers to display with --console-out tickers (default 40)',
    )
    args = parser.parse_args()

    tickers = expand_ticker_args(args.ticker)

    pattern_length = (
        int(args.filter[0]) + 1 if args.filter and args.filter[0].isdigit() else 4
    )

    if args.start:
        start = pd.to_datetime(args.start)
        end = pd.to_datetime(args.end) if args.end else pd.Timestamp.now().normalize()
    else:
        start, end = period_to_start_end(args.period or '1y')
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

    original_start = start
    original_end = end
    cache_tag = f"{original_start.date()}_{original_end.date()}"

    fetch_start = start - pd.Timedelta(days=pattern_length)
    fetch_end = end + pd.Timedelta(days=1)

    total_trades = 0
    total_return_open = 0.0
    total_return_close = 0.0
    total_return_high = 0.0
    total_return_low = 0.0
    total_wins = 0
    rows: list[dict[str, float | int | str]] = []
    trade_rows: list[dict[str, float | str | pd.Timestamp]] = []
    today_buys: list[dict[str, float | str]] = []
    ticker_stats: dict[str, dict[str, float | int | str]] = {}
    is_today_end = original_end.normalize() == pd.Timestamp.now().normalize()

    for ticker in tickers:
        df = fetch_daily_data(ticker, fetch_start, fetch_end, cache_tag)
        if df.empty:
            print(f"No data for {ticker}")
            continue
        trades = backtest_pattern(df, args.filter)
        trades = [
            t
            for t in trades
            if original_start.date() <= t["entry_day"] <= original_end.date()
        ]
        count = len(trades)
        avg_open = (
            sum(t['gain_loss_open_pct'] for t in trades) / count if count else 0.0
        )
        avg_close = (
            sum(t['gain_loss_close_pct'] for t in trades) / count if count else 0.0
        )
        avg_high = (
            sum(t['gain_loss_high_pct'] for t in trades) / count if count else 0.0
        )
        avg_low = (
            sum(t['gain_loss_low_pct'] for t in trades) / count if count else 0.0
        )
        wins = sum(1 for t in trades if t['gain_loss_open_pct'] > 0)
        win_pct = wins / count * 100 if count else 0.0
        loss_pct = (count - wins) / count * 100 if count else 0.0
        backtest_days = df[
            (df["Date"].dt.date >= original_start.date())
            & (df["Date"].dt.date <= original_end.date())
        ]
        total_days = len(backtest_days)
        exec_pct = count / total_days * 100 if total_days else 0.0
        stats_row = {
            'ticker': ticker,
            'trades': count,
            'exec_pct': exec_pct,
            'win_pct': win_pct,
            'loss_pct': loss_pct,
            'avg_open': avg_open,
            'avg_close': avg_close,
            'avg_high': avg_high,
            'avg_low': avg_low,
        }
        ticker_stats[ticker] = stats_row
        if count:
            rows.append(stats_row)
        for trade in trades:
            trade_rows.append({'ticker': ticker, **trade})
        if args.console_out not in ('tickers', 'trades'):
            print(
                f"{ticker}: Trades {count}, Execute {exec_pct:.2f}%, Win {win_pct:.2f}%, "
                f"Loss {loss_pct:.2f}%, Avg Gain/Loss Open {avg_open:.2f}%, "
                f"Close {avg_close:.2f}%, High {avg_high:.2f}%, Low {avg_low:.2f}%"
            )
        total_trades += count
        total_return_open += sum(t['gain_loss_open_pct'] for t in trades)
        total_return_close += sum(t['gain_loss_close_pct'] for t in trades)
        total_return_high += sum(t['gain_loss_high_pct'] for t in trades)
        total_return_low += sum(t['gain_loss_low_pct'] for t in trades)
        total_wins += wins

        if is_today_end and not df[df["Date"] == original_end].empty:
            recent = df[df["Date"] <= original_end].tail(pattern_length - 1)
            if len(recent) == pattern_length - 1:
                days = {f"day{j+2}": recent.iloc[-(j+1)] for j in range(pattern_length - 1)}
                match = False
                if args.filter == "3E":
                    if (
                        days["day4"]["Close"] > days["day4"]["Open"]
                        and days["day3"]["Close"] > days["day3"]["Open"]
                        and days["day2"]["Open"] > days["day3"]["Close"]
                        and days["day2"]["Close"] < days["day3"]["Open"]
                    ):
                        match = True
                elif args.filter == "4E":
                    if (
                        days["day5"]["Close"] > days["day5"]["Open"]
                        and days["day4"]["Close"] > days["day4"]["Open"]
                        and days["day3"]["Close"] > days["day3"]["Open"]
                        and days["day2"]["Open"] > days["day3"]["Close"]
                        and days["day2"]["Close"] < days["day3"]["Open"]
                    ):
                        match = True
                elif args.filter == "3D":
                    if (
                        days["day4"]["Close"]
                        > days["day3"]["Close"]
                        > days["day2"]["Close"]
                    ):
                        match = True
                elif args.filter == "4D":
                    if (
                        days["day5"]["Close"]
                        > days["day4"]["Close"]
                        > days["day3"]["Close"]
                        > days["day2"]["Close"]
                    ):
                        match = True
                if match:
                    price = fetch_current_price(ticker)
                    if price is not None:
                        today_buys.append({"ticker": ticker, "price": price})

    start_label = original_start.strftime('%Y-%m-%d')
    end_label = original_end.strftime('%Y-%m-%d')
    file_label = f"{start_label}-{end_label}-{args.filter}.csv"
    trades_dir = Path('trades') / 'taygetus'
    tickers_dir = Path('tickers') / 'taygetus'
    trades_dir.mkdir(parents=True, exist_ok=True)
    tickers_dir.mkdir(parents=True, exist_ok=True)
    trades_path = trades_dir / file_label
    tickers_path = tickers_dir / file_label

    trades_df = pd.DataFrame(trade_rows)
    trades_df = round_numeric_cols(trades_df)
    trades_df.to_csv(trades_path, index=False)
    if args.console_out == 'trades' and not trades_df.empty:
        if tabulate:
            print(tabulate(trades_df, headers='keys', tablefmt='grid', showindex=False))
        else:
            print(trades_df.to_string(index=False))
    elif args.console_out == 'trades':
        print('No trades to display')
    print(f'Trades saved to {trades_path}')

    tickers_df = pd.DataFrame(
        rows,
        columns=['ticker', 'trades', 'exec_pct', 'win_pct', 'loss_pct', 'avg_open', 'avg_close', 'avg_high', 'avg_low']
    )
    if not tickers_df.empty:
        tickers_df = tickers_df.sort_values(by='avg_open', ascending=False)
        tickers_df = round_numeric_cols(tickers_df)
        tickers_df.to_csv(tickers_path, index=False)
        if args.console_out == 'tickers':
            display_df = tickers_df[
                (tickers_df['win_pct'] > 30) & (tickers_df['exec_pct'] > 2)
            ].head(args.max_out)
            if tabulate:
                print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
            else:
                print(display_df.to_string(index=False))
    else:
        tickers_df.to_csv(tickers_path, index=False)
        if args.console_out == 'tickers':
            print('No tickers to display')
    print(f'Ticker summary saved to {tickers_path}')

    if today_buys:
        print("Today's buys:")
        buy_df = pd.DataFrame(today_buys)
        stats_df = pd.DataFrame([ticker_stats[row['ticker']] for row in today_buys])
        buy_df = buy_df.merge(stats_df, on='ticker')
        buy_df = round_numeric_cols(buy_df)
        if tabulate:
            print(tabulate(buy_df, headers='keys', tablefmt='grid', showindex=False))
        else:
            print(buy_df.to_string(index=False))
        tickers_line = " ".join(t["ticker"].upper() for t in today_buys)
        print(tickers_line)

    if total_trades:
        overall_open = total_return_open / total_trades
        overall_close = total_return_close / total_trades
        overall_high = total_return_high / total_trades
        overall_low = total_return_low / total_trades
        success_rate = total_wins / total_trades * 100
        print(
            f"Overall: Trades {total_trades}, Success Rate {success_rate:.2f}%, "
            f"Average Gain/Loss Open {overall_open:.2f}%, Close {overall_close:.2f}%, "
            f"High {overall_high:.2f}%, Low {overall_low:.2f}%"
        )
    else:
        print("No trades found across tickers")


if __name__ == '__main__':
    main()
