from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def parse_dates(
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[str]]:
    """Normalize incoming date strings for yfinance."""
    if period:
        return None, None, period
    start_dt = pd.to_datetime(start) if start else None
    end_dt = pd.to_datetime(end) if end else None
    return start_dt, end_dt, None
