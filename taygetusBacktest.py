import argparse
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


def backtest_pattern(df: pd.DataFrame) -> list[float]:
    """Return list of percent gains for trades meeting the Taygetus pattern."""
    trades: list[float] = []
    for i in range(3, len(df)):
        day1 = df.iloc[i - 3]
        day2 = df.iloc[i - 2]
        day3 = df.iloc[i - 1]
        day4 = df.iloc[i]

        if (
            day1['Close'] > day1['Open']
            and day2['Close'] > day2['Open']
            and day3['Open'] > day2['Close']
            and day3['Close'] > day2['Open']
        ):
            entry = day3['Close']
            exit = day4['Open']
            pct = (exit - entry) / entry * 100
            trades.append(pct)
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
        choices=['tickers'],
        help='Print per-ticker summary to console in an ASCII table',
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
    rows: list[dict[str, float | int | str]] = []

    for ticker in tickers:
        df = fetch_daily_data(ticker, start, end)
        if df.empty:
            print(f"No data for {ticker}")
            continue
        trades = backtest_pattern(df)
        count = len(trades)
        avg = sum(trades) / count if count else 0.0
        wins = sum(1 for t in trades if t > 0)
        win_pct = wins / count * 100 if count else 0.0
        loss_pct = (count - wins) / count * 100 if count else 0.0
        if args.console_out == 'tickers':
            rows.append(
                {
                    'ticker': ticker,
                    'trades': count,
                    'win_pct': win_pct,
                    'loss_pct': loss_pct,
                    'avg_gain_loss': avg,
                }
            )
        else:
            print(
                f"{ticker}: Trades {count}, Win {win_pct:.2f}%, Loss {loss_pct:.2f}%, Average Gain/Loss {avg:.2f}%"
            )
        total_trades += count
        total_return += sum(trades)

    if args.console_out == 'tickers' and rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(by='avg_gain_loss', ascending=False)
        df = round_numeric_cols(df)
        df = df.head(args.max_out)
        if tabulate:
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        else:
            print(df.to_string(index=False))

    if total_trades:
        overall = total_return / total_trades
        print(f"Overall: Trades {total_trades}, Average Gain/Loss {overall:.2f}%")
    else:
        print("No trades found across tickers")


if __name__ == '__main__':
    main()
