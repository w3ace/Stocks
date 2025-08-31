import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stocks.utils.plots import equity_curve  # noqa: E402


def test_equity_curve_cumulative_change_per_day():
    trades = pd.DataFrame(
        {
            "exit_day": ["2024-01-01", "2024-01-01", "2024-01-03"],
            "gain_loss_pct": [10, -5, 20],
        }
    )

    chart = equity_curve(trades)
    df = pd.DataFrame(chart.data)

    expected_dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    assert list(df["exit_day"]) == list(expected_dates)

    expected_equity = [1.05, 1.05, 1.26]
    assert list(df["equity"].round(2)) == expected_equity
