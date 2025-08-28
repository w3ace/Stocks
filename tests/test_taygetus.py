import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stocks.backtests.taygetus import backtest_pattern  # noqa: E402


def synthetic_df() -> pd.DataFrame:
    data = [
        {"open": 10, "high": 11, "low": 9.5, "close": 11},
        {"open": 11, "high": 12, "low": 10.5, "close": 12},
        {"open": 12, "high": 13, "low": 11.5, "close": 13},
        {"open": 13, "high": 14, "low": 12.5, "close": 14},
        {"open": 14, "high": 14.5, "low": 13, "close": 13},
        {"open": 13, "high": 13.5, "low": 12, "close": 12},
        {"open": 11, "high": 14, "low": 10.5, "close": 13.5},
        {"open": 13.5, "high": 13.6, "low": 13, "close": 13.2},
        {"open": 13.2, "high": 13.3, "low": 13, "close": 13.1},
        {"open": 13.1, "high": 13.2, "low": 12.9, "close": 13.0},
        {"open": 13.0, "high": 13.1, "low": 12.8, "close": 12.9},
        {"open": 12.9, "high": 13.5, "low": 12.8, "close": 13.0},
    ]
    return pd.DataFrame(data)


def test_3e_pattern():
    df = synthetic_df()
    trades = backtest_pattern(df, "3E")
    assert not trades.empty
    assert trades.iloc[0].entry_day == 2
    assert trades.iloc[0].exit_day == 3


def test_3d_pattern():
    df = synthetic_df()
    trades = backtest_pattern(df, "3D")
    assert trades.iloc[0].entry_day == 5
    assert trades.iloc[0].exit_day == 6


def test_3eu_pattern():
    df = synthetic_df()
    trades = backtest_pattern(df, "3EU")
    assert trades.iloc[0].entry_day == 6
    assert trades.iloc[0].exit_day == 7
