import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
import numpy as np
from stock_functions import period_to_start_end, round_numeric_cols
from portfolio_utils import expand_ticker_args
from backtest_filters import (
    DEFAULT_FILTER_ARGS,
    INDICATOR_CHOICES,
    passes_filters,
)
from stocks.backtests.taygetus_run import run_backtest_for_ticker
from stocks.backtests.taygetus import (
    TaygetusPattern,
    backtest_pattern,
    check_pattern,
    check_signal,
    parse_pattern,
)

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




def main() -> None:
    parser = argparse.ArgumentParser(description="Taygetus pattern backtest")
    parser.add_argument('ticker', nargs='+', help='Ticker symbol or list of symbols')
    parser.add_argument(
        '--period',
        help='yfinance period string (e.g. 1y, 6mo) (overrides start/end if both provided)'
    )
    parser.add_argument('--start', help='Start date YYYY-MM-DD')
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
        choices=INDICATOR_CHOICES,
        help=(
            'Indicator filters to enable (default none). '
            'Example: --indicators price atr_pct gap'
        ),
    )
    # === Filtering flags ===
    parser.add_argument(
        "--min-price",
        type=float,
        default=DEFAULT_FILTER_ARGS["min_price"],
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=DEFAULT_FILTER_ARGS["max_price"],
    )
    parser.add_argument(
        "--min-avg-vol",
        dest="min_avg_vol",
        type=float,
        default=DEFAULT_FILTER_ARGS["min_avg_vol"],
    )
    parser.add_argument(
        "--min-dollar-vol",
        dest="min_dollar_vol",
        type=float,
        default=DEFAULT_FILTER_ARGS["min_dollar_vol"],
    )
    parser.add_argument(
        "--min-atr-pct",
        type=float,
        default=DEFAULT_FILTER_ARGS["min_atr_pct"],
    )
    parser.add_argument(
        "--max-atr-pct",
        type=float,
        default=DEFAULT_FILTER_ARGS["max_atr_pct"],
    )
    parser.add_argument(
        "--above-sma",
        type=int,
        choices=[20, 50, 200],
        default=DEFAULT_FILTER_ARGS["above_sma"],
    )
    parser.add_argument(
        "--below-sma",
        type=int,
        choices=[20, 50, 200],
        default=DEFAULT_FILTER_ARGS["below_sma"],
    )
    parser.add_argument(
        "--trend-slope",
        type=float,
        default=DEFAULT_FILTER_ARGS["trend_slope"],
    )  # SMA20 - SMA20_5dago > this
    parser.add_argument(
        "--min-gap-pct",
        type=float,
        default=DEFAULT_FILTER_ARGS["min_gap_pct"],
    )
    parser.add_argument(
        "--body-pct-min",
        dest="body_pct_min",
        type=float,
        default=DEFAULT_FILTER_ARGS["body_pct_min"],
    )
    parser.add_argument(
        "--upper-wick-max",
        dest="upper_wick_max",
        type=float,
        default=DEFAULT_FILTER_ARGS["upper_wick_max"],
    )
    parser.add_argument(
        "--lower-wick-max",
        dest="lower_wick_max",
        type=float,
        default=DEFAULT_FILTER_ARGS["lower_wick_max"],
    )
    parser.add_argument(
        "--pullback-pct-max",
        dest="pullback_pct_max",
        type=float,
        default=DEFAULT_FILTER_ARGS["pullback_pct_max"],
    )
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

    if args.period:
        start, end = period_to_start_end(args.period)
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
    elif args.start:
        start = pd.to_datetime(args.start)
        end = pd.to_datetime(args.end) if args.end else pd.Timestamp.now().normalize()
    else:
        start, end = period_to_start_end('1y')
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

    original_start = start
    original_end = end
    cache_tag = f"{original_start.date()}_{original_end.date()}"

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
        trades, df, bt_args = run_backtest_for_ticker(
            ticker,
            args.pattern,
            original_start,
            original_end,
            args,
            cache_tag,
        )
        if df.empty:
            print(f"No data for {ticker}")
            continue
        indicator_list = getattr(bt_args, "indicators", None)
        print (indicator_list)
        count = len(trades)
        avg_open = trades["open_pct"].mean() if count else 0.0
        avg_close = trades["close_pct"].mean() if count else 0.0
        avg_high = trades["high_pct"].mean() if count else 0.0
        avg_low = trades["low_pct"].mean() if count else 0.0
        wins = int((trades["open_pct"] > 0).sum())
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
        trade_rows.extend(
            trades.assign(ticker=ticker).to_dict("records")
        )
        if args.console_out not in ('tickers', 'trades'):
            print(
                f"{ticker}: Trades {count}, Execute {exec_pct:.2f}%, Win {win_pct:.2f}%, "
                f"Loss {loss_pct:.2f}%, Avg Gain/Loss Open {avg_open:.2f}%, "
                f"Close {avg_close:.2f}%, High {avg_high:.2f}%, Low {avg_low:.2f}%"
            )
        total_trades += count
        total_return_open += trades["open_pct"].sum()
        total_return_close += trades["close_pct"].sum()
        total_return_high += trades["high_pct"].sum()
        total_return_low += trades["low_pct"].sum()
        total_wins += wins

        if is_today_end and not df[df["Date"] == original_end].empty:
            recent = df[df["Date"] <= original_end].tail(pattern_length)
            if len(recent) == pattern_length:
                days = {j + 1: recent.iloc[-(j)] for j in range(pattern_length)}
                if check_pattern(days, pat) and check_signal(days, pat):
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
