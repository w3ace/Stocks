import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stocks.backtests.eldorado import backtest_open_range  # noqa: E402


def synthetic_df() -> pd.DataFrame:
    data = [
        {"open": 100, "high": 101, "low": 99, "close": 102},
        {"open": 102, "high": 103, "low": 100, "close": 101},
        {"open": 101, "high": 102, "low": 99, "close": 98},
    ]
    return pd.DataFrame(data)


def test_backtest_open_range():
    df = synthetic_df()
    trades = backtest_open_range(df, profit_pct=1.0, loss_pct=1.0)
    assert len(trades) == 2
    assert trades.iloc[0]["day"] == 0
    assert trades.iloc[1]["day"] == 2
