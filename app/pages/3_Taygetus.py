import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.components.filters import pattern_selector  # noqa: E402
from app.components.numeric_inputs import slider_with_input  # noqa: E402
from backtest_filters import (  # noqa: E402
    DEFAULT_FILTER_ARGS,
    INDICATOR_CHOICES,
    build_filter_args,
)
from portfolio_utils import expand_ticker_args  # noqa: E402
from stocks.backtests.taygetus_run import run_backtest_for_ticker  # noqa: E402
from stocks.utils.plots import equity_curve, gain_loss_bar  # noqa: E402


@st.cache_data(show_spinner=False)
def run_backtest_cached(
    ticker: str,
    pattern: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    args_dict: dict,
    cache_tag: str,
):
    """Wrapper around :func:`run_backtest_for_ticker` with Streamlit caching.

    ``args_dict`` is converted back to ``argparse.Namespace`` inside the
    function so it can be hashed by Streamlit's cache mechanism.
    """
    args = argparse.Namespace(**args_dict)
    return run_backtest_for_ticker(
        ticker, pattern, start, end, args, cache_tag
    )


st.title("Taygetus Backtest")

tickers_input = st.text_input(
    "Tickers",
    "AAPL",
    help=(
        "Prefix with +<portfolio> to load tickers from the portfolios folder."
    ),
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

with st.expander("Indicator Filters"):
    enabled_ind = st.multiselect("Enable indicators", INDICATOR_CHOICES)
    col_a, col_b = st.columns(2)
    with col_a:
        min_price = slider_with_input(
            "Min Price",
            min_value=0.0,
            max_value=500.0,
            value=float(DEFAULT_FILTER_ARGS["min_price"]),
            step=1.0,
            key="min_price",
        )
        min_avg_vol = st.number_input(
            "Min Avg Vol",
            value=float(DEFAULT_FILTER_ARGS["min_avg_vol"]),
        )
        min_dollar_vol = st.number_input(
            "Min Dollar Vol",
            value=float(DEFAULT_FILTER_ARGS["min_dollar_vol"]),
            step=1.0,
        )
        min_atr_pct = slider_with_input(
            "Min ATR %",
            min_value=0.0,
            max_value=20.0,
            value=float(DEFAULT_FILTER_ARGS["min_atr_pct"]),
            step=0.1,
            key="min_atr_pct",
        )
        above_sma = int(
            st.select_slider(
                "Above SMA",
                options=[20, 50, 200],
                value=int(DEFAULT_FILTER_ARGS["above_sma"]),
            )
        )
        trend_slope = slider_with_input(
            "Trend Slope",
            min_value=-10.0,
            max_value=10.0,
            value=float(DEFAULT_FILTER_ARGS["trend_slope"]),
            step=0.1,
            key="trend_slope",
        )
        body_pct_min = slider_with_input(
            "Body % Min",
            min_value=0.0,
            max_value=100.0,
            value=float(DEFAULT_FILTER_ARGS["body_pct_min"]),
            step=1.0,
            key="body_pct_min",
        )
        upper_wick_max = slider_with_input(
            "Upper Wick % Max",
            min_value=0.0,
            max_value=100.0,
            value=float(DEFAULT_FILTER_ARGS["upper_wick_max"]),
            step=1.0,
            key="upper_wick_max",
        )
        pullback_pct_max = slider_with_input(
            "Pullback % Max",
            min_value=0.0,
            max_value=50.0,
            value=float(DEFAULT_FILTER_ARGS["pullback_pct_max"]),
            step=0.1,
            key="pullback_pct_max",
        )
    with col_b:
        max_price = slider_with_input(
            "Max Price",
            min_value=0.0,
            max_value=500.0,
            value=float(DEFAULT_FILTER_ARGS["max_price"]),
            step=1.0,
            key="max_price",
        )
        max_atr_pct = slider_with_input(
            "Max ATR %",
            min_value=0.0,
            max_value=20.0,
            value=float(DEFAULT_FILTER_ARGS["max_atr_pct"]),
            step=0.1,
            key="max_atr_pct",
        )
        below_sma = int(
            st.select_slider(
                "Below SMA",
                options=[20, 50, 200],
                value=int(DEFAULT_FILTER_ARGS["below_sma"]),
            )
        )
        min_gap_pct = slider_with_input(
            "Min Gap %",
            min_value=0.0,
            max_value=20.0,
            value=float(DEFAULT_FILTER_ARGS["min_gap_pct"]),
            step=0.1,
            key="min_gap_pct",
        )
        lower_wick_max = slider_with_input(
            "Lower Wick % Max",
            min_value=0.0,
            max_value=100.0,
            value=float(DEFAULT_FILTER_ARGS["lower_wick_max"]),
            step=1.0,
            key="lower_wick_max",
        )

    filter_args = build_filter_args(
        indicators=enabled_ind,
        min_price=min_price,
        max_price=max_price,
        min_avg_vol=min_avg_vol,
        min_dollar_vol=min_dollar_vol,
        min_atr_pct=min_atr_pct,
        max_atr_pct=max_atr_pct,
        above_sma=above_sma,
        below_sma=below_sma,
        trend_slope=trend_slope,
        min_gap_pct=min_gap_pct,
        body_pct_min=body_pct_min,
        upper_wick_max=upper_wick_max,
        lower_wick_max=lower_wick_max,
        pullback_pct_max=pullback_pct_max,
    )

if st.button("Run"):
    tokens = [
        t.strip()
        for t in tickers_input.replace(",", " ").split()
        if t.strip()
    ]
    tickers = [t.upper() for t in expand_ticker_args(tokens)]
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(start_str)
    end_ts = pd.Timestamp(end_str)
    cache_tag = f"{start_str}_{end_str}"

    all_trades: list[pd.DataFrame] = []
    ticker_trades: list[tuple[str, pd.DataFrame]] = []

    for ticker in tickers:
        trades, df, _ = run_backtest_cached(
            ticker,
            pattern,
            start_ts,
            end_ts,
            vars(filter_args),
            cache_tag,
        )
        if df.empty:
            st.write(f"No data for {ticker}")
            continue
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
        st.subheader("Trades")
        st.dataframe(combined)
        st.subheader("Equity Curve")
        st.altair_chart(equity_curve(combined), use_container_width=True)
        st.subheader("Gain/Loss")
        st.altair_chart(gain_loss_bar(combined), use_container_width=True)

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
