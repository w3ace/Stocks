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
    data = data[["Open", "Close"]]
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
        One of ``"3E"``, ``"3EC"``, ``"3D"`` or ``"3DC"``. If the first
        character is a digit, it determines the number of pattern days. ``day1``
        refers to the most recent day (exit day) and numbering counts backwards.
    """

    pattern_length = (
        int(filter_value[0]) + 1 if filter_value and filter_value[0].isdigit() else 4
    )

    trades: list[dict[str, float | pd.Timestamp]] = []
    for i in range(pattern_length - 1, len(df)):
        days = {f"day{j + 1}": df.iloc[i - j] for j in range(pattern_length)}

        entry_price = exit_price = None
        if filter_value in {"3E", "3EC"}:
            if (
                days["day4"]["Close"] > days["day4"]["Open"]
                and days["day3"]["Close"] > days["day3"]["Open"]
                and days["day2"]["Open"] > days["day3"]["Close"]
                and days["day2"]["Close"] < days["day3"]["Open"]
            ):
                entry_price = days["day2"]["Close"]
                exit_price = days["day1"]["Close" if filter_value == "3EC" else "Open"]
        elif filter_value in {"3D", "3DC"}:
            if days["day4"]["Close"] > days["day3"]["Close"] > days["day2"]["Close"]:
                entry_price = days["day2"]["Close"]
                exit_price = days["day1"]["Close" if filter_value == "3DC" else "Open"]
        else:
            continue

        if entry_price is not None and exit_price is not None:
            gain_loss = exit_price - entry_price
            gain_loss_pct = gain_loss / entry_price * 100
            trades.append(
                {
                    "entry_day": days["day2"]["Date"].date(),
                    "exit_day": days["day1"]["Date"].date(),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gain_loss": gain_loss,
                    "gain_loss_pct": gain_loss_pct,
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
        choices=['3E', '3EC', '3D', '3DC'],
        default='3E',
        help=(
            'Pattern filter: 3E current Taygetus, 3EC exit at Day4 Close, '
            '3D descending closes, 3DC exit at Day4 Close'
        ),
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
    cache_tag = f"{original_start.date()}_{original_end.date()}_{args.filter}"

    fetch_start = start - pd.Timedelta(days=pattern_length)
    fetch_end = end + pd.Timedelta(days=1)

    total_trades = 0
    total_return = 0.0
    total_wins = 0
    rows: list[dict[str, float | int | str]] = []
    trade_rows: list[dict[str, float | str | pd.Timestamp]] = []
    today_buys: list[dict[str, float | str]] = []
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
        avg = (
            sum(t['gain_loss_pct'] for t in trades) / count if count else 0.0
        )
        wins = sum(1 for t in trades if t['gain_loss_pct'] > 0)
        win_pct = wins / count * 100 if count else 0.0
        loss_pct = (count - wins) / count * 100 if count else 0.0
        if count:
            rows.append(
                {
                    'ticker': ticker,
                    'trades': count,
                    'win_pct': win_pct,
                    'loss_pct': loss_pct,
                    'avg_gain_loss': avg,
                }
            )
        for trade in trades:
            trade_rows.append({'ticker': ticker, **trade})
        if args.console_out not in ('tickers', 'trades'):
            print(
                f"{ticker}: Trades {count}, Win {win_pct:.2f}%, Loss {loss_pct:.2f}%, Average Gain/Loss {avg:.2f}%"
            )
        total_trades += count
        total_return += sum(t['gain_loss_pct'] for t in trades)
        total_wins += wins

        if is_today_end and not df[df["Date"] == original_end].empty:
            recent = df[df["Date"] <= original_end].tail(pattern_length - 1)
            if len(recent) == pattern_length - 1:
                days = {f"day{j+2}": recent.iloc[-(j+1)] for j in range(pattern_length - 1)}
                match = False
                if args.filter in {"3E", "3EC"}:
                    if (
                        days["day4"]["Close"] > days["day4"]["Open"]
                        and days["day3"]["Close"] > days["day3"]["Open"]
                        and days["day2"]["Open"] > days["day3"]["Close"]
                        and days["day2"]["Close"] < days["day3"]["Open"]
                    ):
                        match = True
                elif args.filter in {"3D", "3DC"}:
                    if (
                        days["day4"]["Close"]
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
        columns=['ticker', 'trades', 'win_pct', 'loss_pct', 'avg_gain_loss']
    )
    if not tickers_df.empty:
        tickers_df = tickers_df.sort_values(by='avg_gain_loss', ascending=False)
        tickers_df = round_numeric_cols(tickers_df)
        tickers_df.to_csv(tickers_path, index=False)
        if args.console_out == 'tickers':
            display_df = tickers_df.head(args.max_out)
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
        for row in today_buys:
            print(f"{row['ticker']}: {row['price']:.2f}")

    if total_trades:
        overall = total_return / total_trades
        success_rate = total_wins / total_trades * 100
        print(
            f"Overall: Trades {total_trades}, Success Rate {success_rate:.2f}%, "
            f"Average Gain/Loss {overall:.2f}%"
        )
    else:
        print("No trades found across tickers")


if __name__ == '__main__':
    main()
