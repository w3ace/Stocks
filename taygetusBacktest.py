import argparse
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import yfinance as yf
import numpy as np
from stock_functions import period_to_start_end, round_numeric_cols
from portfolio_utils import expand_ticker_args
from backtest_filters import fetch_daily_data, merge_indicator_data, passes_filters

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - optional dependency
    tabulate = None


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


@dataclass
class TaygetusPattern:
    """Structured representation of a Taygetus pattern string."""

    length: int
    pattern_metric: str
    pattern_dir: str
    signal_metric: str
    signal_dir: str | None = None


def parse_pattern(pattern: str) -> TaygetusPattern:
    """Parse a pattern string into its components.

    Parameters
    ----------
    pattern : str
        Pattern string such as ``"3OUH"``.
    """

    if not pattern or len(pattern) < 4:
        raise ValueError("pattern must be at least 4 characters long")
    length = int(pattern[0])
    patt_metric = pattern[1].upper()
    patt_dir = pattern[2].upper()
    sig_metric = pattern[3].upper()
    sig_dir = pattern[4].upper() if len(pattern) > 4 else None
    if patt_metric not in "OCD" or patt_dir not in "UD":
        raise ValueError("invalid pattern or direction")
    if sig_metric not in "OCDEH":
        raise ValueError("invalid signal")
    if sig_metric in "OCD" and (sig_dir not in "UD"):
        raise ValueError("signal direction required for O/C/D")
    if sig_metric in "EH" and sig_dir and sig_dir not in "UD":
        raise ValueError("invalid signal direction")
    return TaygetusPattern(length, patt_metric, patt_dir, sig_metric, sig_dir)


def _check_pattern(days: dict[int, pd.Series], pat: TaygetusPattern) -> bool:
    """Verify the pattern portion across historical days."""

    num = pat.length
    for k in range(num + 1, 2, -1):
        newer = days[k - 1]
        older = days[k]
        if pat.pattern_metric == "O":
            if pat.pattern_dir == "U" and not (older["Open"] < newer["Open"]):
                return False
            if pat.pattern_dir == "D" and not (older["Open"] > newer["Open"]):
                return False
        elif pat.pattern_metric == "C":
            if pat.pattern_dir == "U" and not (older["Close"] < newer["Close"]):
                return False
            if pat.pattern_dir == "D" and not (older["Close"] > newer["Close"]):
                return False
        elif pat.pattern_metric == "D":
            if pat.pattern_dir == "U" and not (older["Close"] > older["Open"]):
                return False
            if pat.pattern_dir == "D" and not (older["Close"] < older["Open"]):
                return False
    return True


def _check_signal(days: dict[int, pd.Series], pat: TaygetusPattern) -> bool:
    """Check the entry day signal."""

    d2 = days[2]
    d3 = days[3]
    sig = pat.signal_metric
    dir = pat.signal_dir
    if sig == "O": # Open
        if dir == "U":
            return d2["Open"] > d3["Close"]
        else:
            return d2["Open"] < d3["Close"]
    if sig == "C": # Close
        if dir == "U":
            return d2["Close"] > d3["Close"]
        else:
            return d2["Close"] < d3["Close"]
    if sig == "D": # Day
        if dir == "U":
            return d2["Close"] > d2["Open"]
        else:
            return d2["Close"] < d2["Open"]
    if sig == "E": # Engulfing
        bull = d2["Open"] < d3["Close"] and d2["Close"] > d3["Open"]
        bear = d2["Open"] > d3["Close"] and d2["Close"] < d3["Open"]
        if dir == "U":
            return bull
        if dir == "D":
            return bear
        return bull or bear
    if sig == "I":    # Harami
        inside = (
            min(d3["Open"], d3["Close"]) <= d2["Open"] <= max(d3["Open"], d3["Close"])
            and min(d3["Open"], d3["Close"]) <= d2["Close"] <= max(d3["Open"], d3["Close"])
        )
        if not inside:
            return False
        if dir == "U":
            return d2["Close"] > d2["Open"]
        if dir == "D":
            return d2["Close"] < d2["Open"]
        return True
    return False




def backtest_pattern(
    df: pd.DataFrame, pattern: str, args
) -> list[dict[str, float | pd.Timestamp]]:
    """Return detailed trades meeting the selected Taygetus pattern."""

    pat = parse_pattern(pattern)
    pattern_length = pat.length + 1

    trades: list[dict[str, float | pd.Timestamp]] = []
    for i in range(pattern_length - 1, len(df)):
        days = {j + 1: df.iloc[i - j] for j in range(pattern_length)}

        if _check_pattern(days, pat) and _check_signal(days, pat):
            if not args.indicators or passes_filters(df, i, args, args.indicators):
                entry_price = days[2]["Close"]
                exit_open = days[1]["Open"]
                exit_close = days[1]["Close"]
                exit_high = days[1]["High"]
                exit_low = days[1]["Low"]

                gain_open = exit_open - entry_price
                gain_close = exit_close - entry_price
                gain_high = exit_high - entry_price
                gain_low = exit_low - entry_price

                trades.append(
                    {
                        "entry_day": days[2]["Date"].date(),
                        "exit_day": days[1]["Date"].date(),
                        "entry_price": entry_price,
                        "exit_open": exit_open,
                        "open": gain_open,
                        "close": gain_close,
                        "high": gain_high,
                        "low": gain_low,
                        "open_pct": gain_open / entry_price * 100,
                        "close_pct": gain_close / entry_price * 100,
                        "high_pct": gain_high / entry_price * 100,
                        "low_pct": gain_low / entry_price * 100,
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
        '--portfolio-out',
        help='Overwrite portfolio tickers file in ./portfolios/<name>',
    )
    parser.add_argument(
        '--pattern',
        default='3OUH',
        help='Pattern string e.g. 3OUH or 4DUED',
    )
    parser.add_argument(
        '--indicators',
        nargs='+',
        choices=[
            'price',
            'avg_vol',
            'dollar_vol',
            'atr_pct',
            'above_sma',
            'trend_slope',
            'nr7',
            'inside_2',
            'body_pct',
            'upper_wick',
            'lower_wick',
            'pullback_pct',
            'gap',
        ],
        help=(
            'Indicator filters to enable (default none). '
            'Example: --indicators price atr_pct gap'  # previously all enabled by default
        ),
    )
    # === Filtering flags ===
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=200.0)
    parser.add_argument("--min-avg-vol", dest="min_avg_vol", type=float, default=1_000_000)
    parser.add_argument("--min-dollar-vol", dest="min_dollar_vol", type=float, default=20_000_000)
    parser.add_argument("--min-atr-pct", type=float, default=1.0)
    parser.add_argument("--max-atr-pct", type=float, default=8.0)
    parser.add_argument("--above-sma", type=int, choices=[20, 50, 200], default=20)
    parser.add_argument("--trend-slope", type=float, default=0.0)  # SMA20 - SMA20_5dago > this
    parser.add_argument("--min-gap-pct", type=float, default=0.4)
    parser.add_argument("--body-pct-min", dest="body_pct_min", type=float, default=60.0)
    parser.add_argument("--upper-wick-max", dest="upper_wick_max", type=float, default=30.0)
    parser.add_argument("--lower-wick-max", dest="lower_wick_max", type=float, default=40.0)
    parser.add_argument("--pullback-pct-max", dest="pullback_pct_max", type=float, default=6.0)
    parser.add_argument(
        '--max-out',
        type=int,
        default=40,
        help='Maximum tickers to display with --console-out tickers (default 40)',
    )
    args = parser.parse_args()

    tickers = expand_ticker_args(args.ticker)

    pat = parse_pattern(args.pattern)
    pattern_length = pat.length + 1

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
    tickers_line = ""
    ticker_stats: dict[str, dict[str, float | int | str]] = {}
    is_today_end = original_end.normalize() == pd.Timestamp.now().normalize()

    for ticker in tickers:
        df = fetch_daily_data(ticker, fetch_start, fetch_end, cache_tag, "taygetus")
        if df.empty:
            print(f"No data for {ticker}")
            continue
        indicator_list = None
        if args.indicators:
            df, have_ind = merge_indicator_data(df, ticker)
            if have_ind:
                indicator_list = args.indicators
        bt_args = argparse.Namespace(**vars(args))
        bt_args.indicators = indicator_list
        trades = backtest_pattern(df, args.pattern, bt_args)
        trades = [
            t
            for t in trades
            if original_start.date() <= t["entry_day"] <= original_end.date()
        ]
        count = len(trades)
        avg_open = (
            sum(t['open_pct'] for t in trades) / count if count else 0.0
        )
        avg_close = (
            sum(t['close_pct'] for t in trades) / count if count else 0.0
        )
        avg_high = (
            sum(t['high_pct'] for t in trades) / count if count else 0.0
        )
        avg_low = (
            sum(t['low_pct'] for t in trades) / count if count else 0.0
        )
        wins = sum(1 for t in trades if t['open_pct'] > 0)
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
        total_return_open += sum(t['open_pct'] for t in trades)
        total_return_close += sum(t['close_pct'] for t in trades)
        total_return_high += sum(t['high_pct'] for t in trades)
        total_return_low += sum(t['low_pct'] for t in trades)
        total_wins += wins

        if is_today_end and not df[df["Date"] == original_end].empty:
            recent = df[df["Date"] <= original_end].tail(pattern_length)
            if len(recent) == pattern_length:
                days = {j + 1: recent.iloc[-(j + 1)] for j in range(pattern_length)}
                if _check_pattern(days, pat) and _check_signal(days, pat):
                    idx = df.index[df["Date"] == original_end]
                    if len(idx) and (
                        not indicator_list or passes_filters(df, int(idx[0]), bt_args, indicator_list)
                    ):
                        price = fetch_current_price(ticker)
                        if price is not None:
                            today_buys.append({"ticker": ticker, "price": price})


    start_label = original_start.strftime('%Y-%m-%d')
    end_label = original_end.strftime('%Y-%m-%d')
    file_label = f"{start_label}-{end_label}-{args.pattern}.csv"
    trades_dir = Path('trades') / 'taygetus'
    tickers_dir = Path('tickers') / 'taygetus'
    trades_dir.mkdir(parents=True, exist_ok=True)
    tickers_dir.mkdir(parents=True, exist_ok=True)
    trades_path = trades_dir / file_label
    tickers_path = tickers_dir / file_label

    trades_df = pd.DataFrame(trade_rows)
    trades_df = round_numeric_cols(trades_df)
    if "entry_day" in trades_df.columns:
        trades_df = trades_df.sort_values(by="entry_day")
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
        tickers_df = tickers_df.sort_values(by='avg_open', ascending=True)
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
    if args.portfolio_out:
        portfolio_dir = Path("portfolios")
        portfolio_dir.mkdir(exist_ok=True)
        (portfolio_dir / args.portfolio_out).write_text(tickers_line)

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

        # Adjusted summary excluding top/bottom 5% of trades by open_pct
        trim = int(len(trades_df) * 0.05)
        if trim > 0:
            trimmed = trades_df.sort_values("open_pct").iloc[trim:-trim]
        else:
            trimmed = trades_df
        if not trimmed.empty:
            adj_trades = len(trimmed)
            adj_open = trimmed["open_pct"].mean()
            adj_close = trimmed["close_pct"].mean()
            adj_high = trimmed["high_pct"].mean()
            adj_low = trimmed["low_pct"].mean()
            adj_success = (trimmed["open_pct"] > 0).mean() * 100
            print(
                f"Overall Adjusted: Trades {adj_trades}, Success Rate {adj_success:.2f}%, "
                f"Average Gain/Loss Open {adj_open:.2f}%, Close {adj_close:.2f}%, "
                f"High {adj_high:.2f}%, Low {adj_low:.2f}%"
            )
    else:
        print("No trades found across tickers")


if __name__ == '__main__':
    main()
