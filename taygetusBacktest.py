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


def fetch_daily_data(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download daily price data for *ticker* between *start* and *end*."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Open", "Close"]]
    data.reset_index(inplace=True)
    return data


def backtest_pattern(
    df: pd.DataFrame, filter_value: str
) -> list[dict[str, float | pd.Timestamp]]:
    """Return detailed trades meeting the selected Taygetus pattern."""
    trades: list[dict[str, float | pd.Timestamp]] = []
    for i in range(4, len(df)):
        day1 = df.iloc[i - 3]
        day2 = df.iloc[i - 2]
        day3 = df.iloc[i - 1]
        day4 = df.iloc[i]

        entry_price = exit_price = None
        if filter_value == "3E":
            if (
                day1["Close"] > day1["Open"]
                and day2["Close"] > day2["Open"]
                and day3["Open"] > day2["Close"]
                and day3["Close"] < day2["Open"]
            ):
                entry_price = day3["Close"]
                exit_price = day4["Open"]
        elif filter_value == "3D":
            if day1["Close"] > day2["Close"] > day3["Close"]:
                entry_price = day3["Close"]
                exit_price = day4["Open"]
        else:
            continue

        if entry_price is not None and exit_price is not None:
            gain_loss = exit_price - entry_price
            gain_loss_pct = gain_loss / entry_price * 100
            trades.append(
                {
                    "entry_day": day3["Date"].date(),
                    "exit_day": day4["Date"].date(),
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
    parser.add_argument('--end', help='End date YYYY-MM-DD')
    parser.add_argument(
        '--console-out',
        choices=['tickers', 'trades'],
        help='Print per-ticker or per-trade summary to console in an ASCII table',
    )
    parser.add_argument(
        '--filter',
        choices=['3E', '3D'],
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

    if args.start:
        start = pd.to_datetime(args.start)
        end = pd.to_datetime(args.end) if args.end else pd.Timestamp.now()
    else:
        start, end = period_to_start_end(args.period or '1y')

    total_trades = 0
    total_return = 0.0
    total_wins = 0
    rows: list[dict[str, float | int | str]] = []
    trade_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for ticker in tickers:
        df = fetch_daily_data(ticker, start, end)
        if df.empty:
            print(f"No data for {ticker}")
            continue
        trades = backtest_pattern(df, args.filter)
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

    start_label = start.strftime('%Y-%m-%d')
    end_label = end.strftime('%Y-%m-%d')
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
