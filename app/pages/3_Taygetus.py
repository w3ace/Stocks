import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from stocks.backtests.taygetus import backtest_pattern
from stocks.data.fetch import fetch_ticker
from app.components.filters import pattern_selector
from portfolio_utils import expand_ticker_args
from stocks.utils.plots import equity_curve, gain_loss_bar

st.title("Taygetus Backtest")

tickers_input = st.text_input(
    "Tickers",
    "AAPL",
    help="Prefix with +<portfolio> to load tickers from the portfolios folder.",
)
col1, col2, col3, col4 = st.columns(4)
with col1:
    start = st.date_input(
        "Start", dt.date.today() - dt.timedelta(days=30)
    )
with col2:
    end = st.date_input("End", dt.date.today())
with col3:
    period = st.text_input("Period", "1y")
with col4:
    max_out = st.number_input("Max tickers", min_value=1, value=20)
pattern = pattern_selector()

if st.button("Run"):
    tokens = [t.strip() for t in tickers_input.replace(",", " ").split() if t.strip()]
    tickers = [t.upper() for t in expand_ticker_args(tokens)]
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    all_trades: list[pd.DataFrame] = []
    ticker_trades: list[tuple[str, pd.DataFrame]] = []

    for ticker in tickers:
        df = fetch_ticker(
            ticker, start=start_str, end=end_str, period=period or None
        )
        trades = backtest_pattern(df, pattern)
        ticker_trades.append((ticker, trades))
        all_trades.append(trades)

    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        summary_all = {
            "tickers": len(tickers),
            "trades": len(combined),
            "total_gain_pct": float(combined["gain_loss_pct"].sum())
            if not combined.empty
            else 0.0,
        }
        st.subheader("Summary")
        st.write(summary_all)

    for ticker, trades in ticker_trades[: int(max_out)]:
        st.subheader(f"Trades - {ticker}")
        st.dataframe(trades)
        summary = {
            "trades": len(trades),
            "total_gain_pct": float(trades["gain_loss_pct"].sum())
            if not trades.empty
            else 0.0,
        }
        st.write(summary)
        st.subheader("Equity Curve")
        st.altair_chart(equity_curve(trades), use_container_width=True)
        st.subheader("Gain/Loss")
        st.altair_chart(gain_loss_bar(trades), use_container_width=True)

    if len(ticker_trades) > max_out:
        st.write(
            f"Showing first {int(max_out)} of {len(ticker_trades)} tickers."
        )
