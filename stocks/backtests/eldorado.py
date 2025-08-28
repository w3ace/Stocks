from __future__ import annotations

import pandas as pd


def backtest_open_range(
    df: pd.DataFrame, profit_pct: float = 1.0, loss_pct: float = 0.35
) -> pd.DataFrame:
    """Simple open range breakout backtest.

    For each day buy at the open and sell at the close.  A trade is recorded
    when the percentage change exceeds ``profit_pct`` or drops below
    ``-loss_pct``. ``df`` is expected to contain ``open`` and ``close``
    columns.
    """
    df = df.reset_index(drop=True).rename(columns=str.lower)
    trades = []
    for i, row in df.iterrows():
        entry_price = float(row["open"])
        exit_price = float(row["close"])
        change = (exit_price - entry_price) / entry_price * 100
        if change >= profit_pct or change <= -loss_pct:
            trades.append(
                {
                    "day": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gain_loss_pct": change,
                }
            )
    return pd.DataFrame(trades)
