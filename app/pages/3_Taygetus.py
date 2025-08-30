import streamlit as st
from pathlib import Path
import sys

# Ensure project root is on sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from stocks.backtests.taygetus import backtest_pattern
from stocks.data.fetch import fetch_ticker
from stocks.utils.plots import equity_curve, gain_loss_bar
from app.components.filters import pattern_selector

st.title("Taygetus Backtest")

tickers_input = st.text_input("Tickers", "AAPL")
col1, col2, col3 = st.columns(3)
with col1:
    start = st.text_input("Start", "")
with col2:
    end = st.text_input("End", "")
with col3:
    period = st.text_input("Period", "1y")
pattern = pattern_selector()

if st.button("Run"):
    tickers = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]
    for ticker in tickers:
        df = fetch_ticker(ticker, start=start or None, end=end or None, period=period or None)
        trades = backtest_pattern(df, pattern)
        st.subheader(f"Trades - {ticker}")
        st.dataframe(trades)
        summary = {
            "trades": len(trades),
            "total_gain_pct": float(trades["gain_loss_pct"].sum()) if not trades.empty else 0.0,
        }
        st.write(summary)
        st.subheader("Equity Curve")
        st.altair_chart(equity_curve(trades), use_container_width=True)
        st.subheader("Gain/Loss")
        st.altair_chart(gain_loss_bar(trades), use_container_width=True)
