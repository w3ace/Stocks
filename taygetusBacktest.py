import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
import numpy as np
from stock_functions import period_to_start_end, round_numeric_cols
from portfolio_utils import expand_ticker_args
from backtest_filters import fetch_daily_data, add_indicators, passes_filters

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




def backtest_pattern(
    df: pd.DataFrame, filter_value: str, args
) -> list[dict[str, float | pd.Timestamp]]:
    """Return detailed trades meeting the selected Taygetus pattern.

    Parameters
    ----------
    df : pd.DataFrame
        Daily price data.
    filter_value : str
        One of ``"3E"``, ``"4E"``, ``"3D"`` or ``"4D"``. If the first
        character is a digit, it determines the number of pattern days. ``1``
        refers to the most recent day (exit day) and numbering counts backwards.
    args : argparse.Namespace
        Command line arguments containing filter settings.

    Notes
    -----
    Trades are entered at day 2 close and evaluated at day 1 open, close,
    high and low. Gain/loss percentages are calculated for each of these
    potential exits.
    """

    pattern_length = (
        int(filter_value[0]) + 1 if filter_value and filter_value[0].isdigit() else 4
    )

    trades: list[dict[str, float | pd.Timestamp]] = []
    for i in range(pattern_length - 1, len(df)):
        days = {j + 1: df.iloc[i - j] for j in range(pattern_length)}

        entry_price = None
        letter = filter_value[1:]           # "E" or "D"
        num = int(filter_value[0])          # 3 or 4

        if letter == "E":
            # Consecutive green candles on days (num+1 .. 3), e.g. for 3E: day4, day3
            up_ok = all(
                days[k]["Close"] > days[k]["Open"]
                for k in range(num + 1, 2, -1)
            )
            # Gap up + red on day2 (your original condition)
            gap_red_ok = (
                days[2]["Open"] > days[3]["Close"] and
                days[2]["Close"] < days[3]["Open"]
            )
            if up_ok and gap_red_ok:
                entry_price = days[2]["Close"]

        elif letter == "F":
            # Consecutive green candles on days (num+1 .. 3), e.g. for 3E: day4, day3
            down_ok = all(
                days[k]["Close"] < days[k]["Open"]
                for k in range(num + 1, 2, -1)
            )
            # Gap up + red on day2 (your original condition)
            gap_green_ok = (
                days[2]["Open"] < days[3]["Close"] and
                days[2]["Close"] > days[3]["Open"]
            )
            if down_ok and gap_green_ok:
                entry_price = days[2]["Close"]

        elif letter == "D":
            # Strictly decreasing closes across (num+1 .. 2), e.g. for 3D: day4 > day3 > day2
            down_ok = all(
                days[k]["Close"] > days[k - 1]["Close"]
                for k in range(num + 1, 1, -1)
            )
            if down_ok:
                entry_price = days[2]["Close"]

        elif letter == "U":
            # Strictly decreasing closes across (num+1 .. 2), e.g. for 3D: day4 > day3 > day2
            down_ok = all(
                days[k]["Close"] > days[k - 1]["Close"]
                for k in range(num + 1, 1, -1)
            )
            if down_ok:
                entry_price = days[2]["Close"]
        else:
            # Unrecognized filter
            pass

        if entry_price is not None and (
            not args.indicators or passes_filters(df, i, args, args.indicators)
        ):
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
        '--filter',
        choices=['3E', '4E', '3D', '4D', '5D','2F','3F', '4F', '5F', '6F', '3U', '4U', '5U', '6U'],
        default='3E',
        help='Pattern filter: 3E current Taygetus, 3D descending closes',
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
        df = fetch_daily_data(ticker, fetch_start, fetch_end, cache_tag, "taygetus")
        if df.empty:
            print(f"No data for {ticker}")
            continue
        if args.indicators:
            df = add_indicators(df)
        trades = backtest_pattern(df, args.filter, args)
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
            recent = df[df["Date"] <= original_end].tail(pattern_length - 1)
            if len(recent) == pattern_length - 1:
                days = {j + 2: recent.iloc[-(j + 1)] for j in range(pattern_length - 1)}
                match = False
            if args.filter in {"3E", "4E"}:
                num_up_days = int(args.filter[0])  # 3 or 4
                # Check consecutive green candles
                if all(days[n]["Close"] > days[n]["Open"] for n in range(num_up_days + 1, 1, -1)):
                    # Gap up and red candle on last day before entry
                    if days[2]["Open"] > days[3]["Close"] and days[2]["Close"] < days[3]["Open"]:
                        match = True

            elif args.filter in {"2F", "3F", "4F"}:
                num_down_days = int(args.filter[0])  # 3 or 4
                # Check consecutive green candles
                if all(days[n]["Close"] < days[n]["Open"] for n in range(num_down_days + 1, 1, -1)):
                    # Gap up and red candle on last day before entry
                    if days[2]["Open"] < days[3]["Close"] and days[2]["Close"] > days[3]["Open"]:
                        match = True

            elif args.filter in {"3D", "4D", "5D"}:
                num_down_days = int(args.filter[0])  # 3 or 4
                # Check consecutive closes decreasing
                if all(days[n]["Close"] > days[n - 1]["Close"] for n in range(num_down_days + 1, 2, -1)):
                    match = True

            # Reuse filter gate with the true df index (1 == original_end)
            if match:
                idx = df.index[df["Date"] == original_end]
                if len(idx) and (
                    not args.indicators or passes_filters(df, int(idx[0]), args, args.indicators)
                ):
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
