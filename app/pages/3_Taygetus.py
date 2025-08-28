import streamlit as st

from stocks.backtests.taygetus import backtest_pattern
from stocks.data.fetch import fetch_ticker
from stocks.utils.plots import equity_curve, gain_loss_bar
from app.components.filters import pattern_selector

st.title("Taygetus Backtest")

ticker = st.text_input("Ticker", "AAPL")
col1, col2, col3 = st.columns(3)
with col1:
    start = st.text_input("Start", "")
with col2:
    end = st.text_input("End", "")
with col3:
    period = st.text_input("Period", "1y")
pattern = pattern_selector()

if st.button("Run"):
    df = fetch_ticker(ticker, start=start or None, end=end or None, period=period or None)
    trades = backtest_pattern(df, pattern)
    st.subheader("Trades")
    st.dataframe(trades)
    st.subheader("Summary")
    summary = {
        "trades": len(trades),
        "total_gain_pct": float(trades["gain_loss_pct"].sum()) if not trades.empty else 0.0,
    }
    st.write(summary)
    st.subheader("Equity Curve")
    st.altair_chart(equity_curve(trades), use_container_width=True)
    st.subheader("Gain/Loss")
    st.altair_chart(gain_loss_bar(trades), use_container_width=True)
