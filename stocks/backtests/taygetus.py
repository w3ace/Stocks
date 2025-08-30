from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

# === Legacy simple pattern backtest ===


def prepare_days(df_row_indexed: pd.DataFrame) -> List[Dict[str, float]]:
    """Return a list of OHLC dictionaries from *df_row_indexed*.

    This helper is used by the legacy backtest implementation which supports
    simple patterns such as ``"3E"`` or ``"3EU"``.
    The function is retained for backwards compatibility with existing
    tests and callers.
    """

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


def _backtest_pattern_simple(
    df: pd.DataFrame, filter_value: str
) -> pd.DataFrame:
    """Legacy backtester supporting the original simple pattern syntax."""

    df = df.reset_index(drop=True).rename(columns=str.lower)
    match = re.match(r"(\d+)", filter_value)
    n = int(match.group(1)) if match else 0
    trades: List[Dict[str, float]] = []
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
                    "gain_loss_pct": (exit_price - entry_price)
                    / entry_price
                    * 100,
                }
            )
    columns = [
        "entry_day",
        "exit_day",
        "entry_price",
        "exit_price",
        "gain_loss_pct",
    ]
    return pd.DataFrame(trades, columns=columns)


# === New Taygetus pattern engine ===


@dataclass
class TaygetusPattern:
    """Structured representation of a Taygetus pattern string."""

    length: int
    pattern_metric: str
    pattern_dir: str
    signal_metric: str
    signal_dir: Optional[str] = None


def parse_pattern(pattern: str) -> TaygetusPattern:
    """Parse an advanced pattern string into its components."""

    if not pattern or len(pattern) < 4:
        raise ValueError("pattern must be at least 4 characters long")
    length = int(pattern[0])
    patt_metric = pattern[1].upper()
    patt_dir = pattern[2].upper()
    sig_metric = pattern[3].upper()
    sig_dir = pattern[4].upper() if len(pattern) > 4 else None
    if patt_metric not in "OCD" or patt_dir not in "UD":
        raise ValueError("invalid pattern or direction")
    if sig_metric not in "OCDEI":
        raise ValueError("invalid signal")
    if sig_metric in "OCD" and (sig_dir not in "UD"):
        raise ValueError("signal direction required for O/C/D")
    if sig_metric in "EI" and sig_dir and sig_dir not in "UD":
        raise ValueError("invalid signal direction")
    return TaygetusPattern(length, patt_metric, patt_dir, sig_metric, sig_dir)


def check_pattern(days: Dict[int, pd.Series], pat: TaygetusPattern) -> bool:
    """Verify the pattern portion across historical days."""

    num = pat.length
    for k in range(num + 1, 2, -1):
        newer = days[k - 1]
        older = days[k]
        if pat.pattern_metric == "O":
            if pat.pattern_dir == "U" and not (older["Open"] < newer["Open"]):
                return False
            if pat.pattern_dir == "D" and not (older["Open"] > newer["Open"]):
                return False
        elif pat.pattern_metric == "C":
            if (
                pat.pattern_dir == "U"
                and not (older["Close"] < newer["Close"])
            ):
                return False
            if (
                pat.pattern_dir == "D"
                and not (older["Close"] > newer["Close"])
            ):
                return False
        elif pat.pattern_metric == "D":
            if pat.pattern_dir == "U" and not (older["Close"] > older["Open"]):
                return False
            if pat.pattern_dir == "D" and not (older["Close"] < older["Open"]):
                return False
    return True


def check_signal(days: Dict[int, pd.Series], pat: TaygetusPattern) -> bool:
    """Check the entry day signal."""

    d2 = days[2]
    d3 = days[3]
    sig = pat.signal_metric
    direction = pat.signal_dir
    if sig == "O":  # Open
        if direction == "U":
            return d2["Open"] > d3["Close"]
        else:
            return d2["Open"] < d3["Close"]
    if sig == "C":  # Close
        if direction == "U":
            return d2["Close"] > d3["Close"]
        else:
            return d2["Close"] < d3["Close"]
    if sig == "D":  # Day
        if direction == "U":
            return d2["Close"] > d2["Open"]
        else:
            return d2["Close"] < d2["Open"]
    if sig == "E":  # Engulfing
        bull = d2["Open"] < d3["Close"] and d2["Close"] > d3["Open"]
        bear = (
            d2["Open"] > d3["Close"]
            and d2["Close"] < d3["Open"]
        )
        if direction == "U":
            return bull
        if direction == "D":
            return bear
        return bull or bear
    if sig == "I":  # Harami
        lo = min(d3["Open"], d3["Close"])
        hi = max(d3["Open"], d3["Close"])
        inside = lo <= d2["Open"] <= hi and lo <= d2["Close"] <= hi
        if not inside:
            return False
        if direction == "U":
            return d2["Close"] > d2["Open"]
        if direction == "D":
            return d2["Close"] < d2["Open"]
        return True
    return False


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with standard columns used by the new backtester."""

    out = df.copy()
    if "Date" not in out.columns:
        out = out.reset_index()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    rename_map = {c: c.title() for c in out.columns}
    out = out.rename(columns=rename_map)
    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def _backtest_pattern_advanced(
    df: pd.DataFrame, pattern: str, args: Optional[object] = None
) -> pd.DataFrame:
    """Advanced Taygetus pattern backtest supporting rich pattern syntax."""

    pat = parse_pattern(pattern)
    pattern_length = pat.length + 1
    df = _prepare_dataframe(df)
    trades: List[Dict[str, float]] = []
    for i in range(pattern_length - 1, len(df)):
        days = {j + 1: df.iloc[i - j] for j in range(pattern_length)}
        if check_pattern(days, pat) and check_signal(days, pat):
            if args and getattr(args, "indicators", None):
                try:
                    from backtest_filters import passes_filters  # type: ignore
                except Exception:  # pragma: no cover - optional dependency
                    passes_filters = None
                if passes_filters and not passes_filters(
                    df, i, args, args.indicators
                ):
                    continue
            entry_price = days[2]["Close"]
            exit_open = days[1]["Open"]
            exit_close = days[1]["Close"]
            exit_high = days[1]["High"]
            exit_low = days[1]["Low"]
            gain_open = exit_open - entry_price
            gain_close = exit_close - entry_price
            gain_high = exit_high - entry_price
            gain_low = exit_low - entry_price
            trades.append(
                {
                    "entry_day": days[2]["Date"].date(),
                    "exit_day": days[1]["Date"].date(),
                    "entry_price": entry_price,
                    "exit_open": exit_open,
                    "exit_close": exit_close,
                    "exit_high": exit_high,
                    "exit_low": exit_low,
                    "open": gain_open,
                    "close": gain_close,
                    "high": gain_high,
                    "low": gain_low,
                    "open_pct": gain_open / entry_price * 100,
                    "close_pct": gain_close / entry_price * 100,
                    "high_pct": gain_high / entry_price * 100,
                    "low_pct": gain_low / entry_price * 100,
                    "gain_loss_pct": gain_close / entry_price * 100,
                }
            )
    columns = [
        "entry_day",
        "exit_day",
        "entry_price",
        "exit_open",
        "exit_close",
        "exit_high",
        "exit_low",
        "open",
        "close",
        "high",
        "low",
        "open_pct",
        "close_pct",
        "high_pct",
        "low_pct",
        "gain_loss_pct",
    ]
    return pd.DataFrame(trades, columns=columns)


def _is_advanced(pattern: str) -> bool:
    """Return True if *pattern* uses the advanced Taygetus syntax."""

    return (
        len(pattern) >= 4
        and pattern[1].upper() in "OCD"
        and pattern[2].upper() in "UD"
    )


def backtest_pattern(
    df: pd.DataFrame, pattern: str, args: Optional[object] = None
) -> pd.DataFrame:
    """Backtest *pattern* over *df* using the appropriate engine.

    ``pattern`` may be specified using either the legacy syntax (e.g.
    ``"3E"``) or the advanced Taygetus syntax (e.g. ``"3OUH"``).  The
    function dispatches to the correct implementation based on the pattern
    format and always returns a ``pandas.DataFrame`` of trades.
    """

    if _is_advanced(pattern):
        return _backtest_pattern_advanced(df, pattern, args)
    return _backtest_pattern_simple(df, pattern)


__all__ = [
    "backtest_pattern",
    "pattern_match",
    "prepare_days",
    "parse_pattern",
    "check_pattern",
    "check_signal",
    "TaygetusPattern",
]
