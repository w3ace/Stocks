import importlib.util
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
MODULE_PATH = REPO_ROOT / "gapDetector.py"
SPEC = importlib.util.spec_from_file_location("gapDetector", MODULE_PATH)
gapDetector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gapDetector)


def test_analyze_gaps_reports_intraday_extremes(monkeypatch):
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Open": [100.0, 110.0, 90.0],
            "High": [101.0, 121.0, 95.0],
            "Low": [99.0, 105.0, 81.0],
            "Close": [100.0, 115.0, 85.0],
        }
    )

    def fake_fetch_daily_data(ticker, start, end):
        return data

    monkeypatch.setattr(gapDetector, "fetch_daily_data", fake_fetch_daily_data)

    result = gapDetector.analyze_gaps(
        "FAKE",
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        tolerance=0,
    )

    assert result["gap_up_days"] == 1
    assert result["gap_down_days"] == 1
    assert result["avg_max_up_pct"] == 10.0
    assert result["avg_max_down_pct"] == -10.0
