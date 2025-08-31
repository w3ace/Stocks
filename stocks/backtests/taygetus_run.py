from __future__ import annotations

import argparse

import pandas as pd

from backtest_filters import fetch_daily_data, merge_indicator_data
from .taygetus import backtest_pattern, parse_pattern


def run_backtest_for_ticker(
    ticker: str,
    pattern: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    args: argparse.Namespace,
    cache_tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame, argparse.Namespace]:
    """Run the Taygetus backtest for ``ticker`` between ``start`` and ``end``.

    Parameters
    ----------
    ticker : str
        Symbol to backtest.
    pattern : str
        Pattern string understood by :func:`backtest_pattern`.
    start, end : pd.Timestamp
        Date range to evaluate.  ``start`` and ``end`` correspond to the
        entry day and are *not* adjusted for pattern lookback.
    args : argparse.Namespace
        Indicator filter arguments.  ``args.indicators`` may be ``None`` or a
        list of enabled indicator names.
    cache_tag : str
        Tag used by :func:`fetch_daily_data` to cache downloads.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, argparse.Namespace]
        ``(trades, df, bt_args)`` where ``trades`` contains the executed trades,
        ``df`` is the price data with any indicators merged in and ``bt_args``
        is the ``argparse.Namespace`` passed to ``backtest_pattern`` (with the
        ``indicators`` attribute adjusted based on dataset availability).
    """

    pat = parse_pattern(pattern)
    fetch_start = start - pd.Timedelta(days=pat.length + 1)
    fetch_end = end + pd.Timedelta(days=1)
    df = fetch_daily_data(ticker, fetch_start, fetch_end, cache_tag, "taygetus")
    if df.empty:
        return pd.DataFrame(), df, args
    indicator_list = None
    if getattr(args, "indicators", None):
        df, have_ind = merge_indicator_data(df, ticker)
        if have_ind:
            indicator_list = args.indicators
    bt_args = argparse.Namespace(**vars(args))
    bt_args.indicators = indicator_list
    trades = backtest_pattern(df, pattern, bt_args)
    trades = trades[
        (trades["entry_day"] >= start.date()) & (trades["entry_day"] <= end.date())
    ]
    return trades, df, bt_args
