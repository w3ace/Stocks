from datetime import datetime, timedelta


def period_to_start_end(period: str, end: datetime | None = None) -> tuple[datetime, datetime]:
    """Convert a yfinance period string to explicit ``start`` and ``end`` dates.

    Parameters
    ----------
    period : str
        Period string understood by :func:`yfinance.download`.
    end : datetime, optional
        End date for the range.  ``None`` uses the current day in UTC.

    Returns
    -------
    (datetime, datetime)
        Tuple of ``(start, end)`` dates.
    """
    end = end or datetime.utcnow()

    period_map = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "7d": timedelta(days=7),
        "14d": timedelta(days=14),
        "30d": timedelta(days=30),
        "60d": timedelta(days=60),
        "90d": timedelta(days=90),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
        "5y": timedelta(days=1825),
        "10y": timedelta(days=3650),
        "ytd": None,
        "max": None,
    }

    p = period.lower()
    if p == "ytd":
        start = datetime(end.year, 1, 1)
    elif p == "max":
        start = datetime(1900, 1, 1)
    else:
        delta = period_map.get(p)
        if delta is None:
            # Default to one year if unknown
            delta = timedelta(days=365)
        start = end - delta

    return start, end

def choose_yfinance_interval(start=None, end=None, period=None):
    """
    Chooses the best yfinance interval based on either start/end or period.
    Returns a string like '1m', '5m', '1d', etc.
    """
    
    # Interval tiers (from yfinance limits)
    intervals = [
        ("1m", timedelta(days=7)),
        ("2m", timedelta(days=60)),
        ("5m", timedelta(days=60)),
        ("15m", timedelta(days=60)),
        ("30m", timedelta(days=60)),
        ("60m", timedelta(days=730)),  # 2 years
        ("1d", timedelta(days=5000)),  # ~13 years or more
        ("1wk", timedelta(days=15000)), # ~41 years
        ("1mo", timedelta(days=25000))  # ~68 years
    ]

    # Calculate duration
    if period:
        # Use predefined periods
        period_map = {
            "1d": timedelta(days=1),
            "5d": timedelta(days=5),
            "7d": timedelta(days=7),
            "14d": timedelta(days=14),
            "30d": timedelta(days=30),
            "60d": timedelta(days=60),
            "90d": timedelta(days=90),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
            "5y": timedelta(days=1825),
            "10y": timedelta(days=3650),
            "ytd": timedelta(days=180),  # estimate
            "max": timedelta(days=25000)
        }
        duration = period_map.get(period.lower(), timedelta(days=365))  # default to 1y if unknown
    elif start and end:
        if isinstance(start, str): start = datetime.fromisoformat(start)
        if isinstance(end, str): end = datetime.fromisoformat(end)
        duration = end - start
    else:
        raise ValueError("You must provide either a period or both start and end dates.")

    # Pick the longest interval that supports the requested duration
    for interval, max_duration in intervals:
        if duration <= max_duration:
            return interval

    return "1mo"  # fallback for extremely long durations
