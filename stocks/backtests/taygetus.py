from __future__ import annotations

import re
from typing import List, Dict

import pandas as pd


def prepare_days(df_row_indexed: pd.DataFrame) -> List[Dict[str, float]]:
    """Return a list of OHLC dictionaries from *df_row_indexed*."""
    return df_row_indexed[["open", "high", "low", "close"]].to_dict("records")


def pattern_match(days: List[Dict[str, float]], filter_value: str) -> bool:
    """Return True if *days* satisfy *filter_value* pattern."""
    match = re.match(r"(\d+)([A-Za-z]+)", filter_value)
    if not match:
        return False
    n, suffix = int(match.group(1)), match.group(2).upper()
    if len(days) < n:
        return False
    closes = [d["close"] for d in days]
    if suffix == "E":
        return all(closes[i] < closes[i + 1] for i in range(n - 1))
    if suffix == "D":
        return all(closes[i] > closes[i + 1] for i in range(n - 1))
    if suffix == "EU":
        down = all(closes[i] > closes[i + 1] for i in range(n - 2))
        prev = days[-2]
        cur = days[-1]
        engulf = (
            cur["open"] < prev["close"] and cur["close"] > prev["open"]
        )
        return down and engulf
    return False


def backtest_pattern(df: pd.DataFrame, filter_value: str) -> pd.DataFrame:
    """Backtest *filter_value* over *df* and return trades."""
    df = df.reset_index(drop=True).rename(columns=str.lower)
    match = re.match(r"(\d+)", filter_value)
    n = int(match.group(1)) if match else 0
    trades = []
    for i in range(n - 1, len(df) - 1):
        window = df.iloc[i - n + 1:i + 1]
        days = prepare_days(window)
        if pattern_match(days, filter_value):
            entry_price = df.loc[i, "close"]
            exit_price = df.loc[i + 1, "close"]
            trades.append(
                {
                    "entry_day": i,
                    "exit_day": i + 1,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gain_loss_pct": (
                        (exit_price - entry_price) / entry_price * 100
                    ),
                }
            )
    return pd.DataFrame(trades)
