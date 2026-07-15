import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

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

    def fake_fetch_daily_data(ticker, start, end, cache_tag):
        return data

    monkeypatch.setattr(gapDetector, "fetch_daily_data", fake_fetch_daily_data)

    result = gapDetector.analyze_gaps(
        "FAKE",
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        tolerance=0,
        success=0,
        cache_tag="test",
    )

    assert result["gap_up_days"] == 1
    assert result["gap_down_days"] == 1
    assert result["avg_max_up_pct"] == 10.0
    assert result["avg_max_down_pct"] == -10.0
    assert result["success_up_close_pct"] == 100.0
    assert result["success_up_max_reward_pct"] == 100.0
    assert result["success_up_pct"] == 100.0
    assert result["success_down_close_pct"] == 100.0
    assert result["success_down_max_reward_pct"] == 100.0
    assert result["success_down_pct"] == 100.0
    assert result["success_both_close_pct"] == 100.0
    assert result["success_both_max_reward_pct"] == 100.0
    assert result["success_both_pct"] == 100.0
    assert result["risk_reward_up_close"] == 1.0
    assert result["risk_reward_up_max_reward"] == pytest.approx(2.2)
    assert result["risk_reward_down_close"] == 1.0
    assert result["risk_reward_down_max_reward"] == pytest.approx(1.8)


def test_fetch_current_extended_gap_uses_latest_extended_price(monkeypatch):
    history = pd.DataFrame(
        {"Close": [100.0, 102.0, 105.0]},
        index=pd.to_datetime(
            [
                "2024-01-02 15:59",
                "2024-01-02 16:00",
                "2024-01-02 16:30",
            ]
        ),
    )

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **kwargs):
            assert kwargs["prepost"] is True
            return history

    monkeypatch.setattr(gapDetector.yf, "Ticker", FakeTicker)

    result = gapDetector.fetch_current_extended_gap("FAKE")

    assert result["ticker"] == "FAKE"
    assert result["current_gap_pct"] == 2.941176470588235
    assert result["current_gap_direction"] == "up"


def test_current_gap_tickers_filters_by_direction_and_tolerance(monkeypatch):
    gaps = {
        "UP": {"ticker": "UP", "current_gap_pct": 2.0},
        "DOWN": {"ticker": "DOWN", "current_gap_pct": -1.5},
        "FLAT": {"ticker": "FLAT", "current_gap_pct": 0.5},
    }

    monkeypatch.setattr(
        gapDetector, "fetch_current_extended_gap", lambda ticker: gaps[ticker]
    )
    monkeypatch.setattr(gapDetector, "fetch_daily_data", lambda *args: pd.DataFrame())
    monkeypatch.setattr(gapDetector, "calculate_atr", lambda data: (0.0, 1.0))
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-31")

    assert list(
        gapDetector.current_gap_tickers(
            ["UP", "DOWN", "FLAT"], 1.0, "up", start, end, "test"
        )
    ) == [
        "UP",
    ]
    assert list(
        gapDetector.current_gap_tickers(
            ["UP", "DOWN", "FLAT"], 1.0, "down", start, end, "test"
        )
    ) == [
        "DOWN",
    ]
