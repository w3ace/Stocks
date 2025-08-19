import pandas as pd
import yfinance as yf
import numpy as np
from pathlib import Path

# Opt-in to pandas' future behavior to avoid silent downcasting warnings
pd.set_option("future.no_silent_downcasting", True)

CACHE_DIR = Path(__file__).resolve().parent / "yfinance_cache"
INDICATOR_DIR = Path(__file__).resolve().parent / "datasets" / "indicators"


def fetch_daily_data(ticker: str, start: pd.Timestamp, end: pd.Timestamp, cache_tag: str, subdir: str) -> pd.DataFrame:
    """Download daily price data for *ticker* between *start* and *end*.

    Cached under ``yfinance_cache/subdir/cache_tag`` grouped by first letter of ticker.
    Caching disabled for current day before 4:30pm US/Eastern.
    """
    now_est = pd.Timestamp.now(tz="US/Eastern")
    four_thirty = pd.Timestamp("16:30", tz="US/Eastern").time()
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    cache_enabled = not (start_date <= now_est.date() < end_date and now_est.time() < four_thirty)
    cache_file = CACHE_DIR / subdir / cache_tag / ticker[0].upper() / ticker
    if cache_enabled and cache_file.exists():
        try:
            data = pd.read_pickle(cache_file)
        except Exception:
            data = pd.DataFrame()
    else:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if cache_enabled and not data.empty:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                data.to_pickle(cache_file)
            except Exception:
                pass
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.reset_index(inplace=True)
    return data


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling indicators used by filters."""
    if df.empty:
        return df
    df = df.copy()
    df["PrevClose"] = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["PrevClose"]).abs()
    tr3 = (df["Low"] - df["PrevClose"]).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    df["ATR14"] = tr.rolling(14).mean()
    df["ATRpct"] = (df["ATR14"] / df["Close"]) * 100.0
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["SMA20_5dago"] = df["SMA20"].shift(5)
    df["VolSMA20"] = df["Volume"].rolling(20).mean()
    df["DollarVol20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    body = (df["Close"] - df["Open"]).abs()
    df["BodyPct"] = (body / rng) * 100.0
    df["UpperWickPct"] = ((df["High"] - df[["Open", "Close"]].max(axis=1)) / rng) * 100.0
    df["LowerWickPct"] = ((df[["Open", "Close"]].min(axis=1) - df["Low"]) / rng) * 100.0
    df["DayRange"] = df["High"] - df["Low"]
    df["NR7"] = df["DayRange"] < df["DayRange"].rolling(7).max().shift(1)
    df["Inside"] = (df["High"] <= df["High"].shift(1)) & (df["Low"] >= df["Low"].shift(1))
    ins = df["Inside"].astype("boolean")
    df["Inside2"] = ins & ins.shift(1, fill_value=False)
    hh20 = df["Close"].rolling(20).max()
    df["PullbackPct20"] = ((hh20 - df["Close"]) / hh20) * 100.0
    return df


def merge_indicator_data(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, bool]:
    """Attach pre-computed indicator data for ``ticker`` to ``df``.

    The indicator data is read from ``datasets/indicators/<ticker>.csv``. If the
    file is missing, a warning is printed and the original ``df`` is returned
    unchanged along with ``False`` to signal that indicators are unavailable.
    When the latest day in ``df`` is newer than the indicator dataset, the most
    recent indicator values are forward-filled to cover the gap.

    Parameters
    ----------
    df : pd.DataFrame
        Daily price data.
    ticker : str
        Ticker symbol used to locate the indicator dataset.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        The merged DataFrame and a flag indicating whether indicators were
        successfully loaded.
    """
    dest = INDICATOR_DIR / f"{ticker}.csv"
    if not dest.exists():
        print(f"Warning: indicator file not found for {ticker} at {dest}. Skipping indicators.")
        return df, False
    try:
        ind = pd.read_csv(dest, parse_dates=["Date"])
    except Exception:
        print(f"Warning: failed to read indicator file for {ticker} at {dest}. Skipping indicators.")
        return df, False
    if ind.empty:
        print(f"Warning: indicator file for {ticker} is empty. Skipping indicators.")
        return df, False
    ind = ind.sort_values("Date")
    # Drop raw price columns so merging doesn't create duplicate
    # ``_x``/``_y`` suffixed columns which would otherwise cause a
    # ``KeyError`` when we forward fill indicator values.  The price
    # data from ``df`` is authoritative and already contains these
    # fields.
    price_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    ind = ind.drop(columns=[c for c in price_cols if c in ind.columns], errors="ignore")
    df = df.sort_values("Date")
    merged = df.merge(ind, on="Date", how="left")
    cols = [c for c in ind.columns if c != "Date"]
    if cols:
        merged[cols] = merged[cols].ffill().infer_objects(copy=False)
    return merged, True


def passes_filters(
    df: pd.DataFrame, i: int, args, enabled: list[str] | set[str] | None = None
) -> bool:
    """Apply selected indicator filters to ``day2`` (``i-1``).

    Parameters
    ----------
    df : pd.DataFrame
        Daily price data with indicators.
    i : int
        Index of the exit day (day1). ``day2`` is ``i-1``.
    args : argparse.Namespace
        Command line arguments containing filter settings.
    enabled : Iterable[str] | None
        Names of indicator filters to enforce. If ``None`` or empty, no
        indicator-based filtering is applied.
    """
    if not enabled:
        return True
    enabled = set(enabled)
    if i - 2 < 0:
        return False
    d2 = df.iloc[i - 1]
    d3 = df.iloc[i - 2]
    if "price" in enabled and not (args.min_price <= d2["Close"] <= args.max_price):
        return False
    if "avg_vol" in enabled and (pd.isna(d2.get("VolSMA20")) or d2["VolSMA20"] < args.min_avg_vol):
        return False
    if "dollar_vol" in enabled and (pd.isna(d2.get("DollarVol20")) or d2["DollarVol20"] < args.min_dollar_vol):
        return False
    if "atr_pct" in enabled and (
        pd.isna(d2.get("ATRpct")) or not (args.min_atr_pct <= d2["ATRpct"] <= args.max_atr_pct)
    ):
        return False
    if "nr7" in enabled and not bool(d2.get("NR7", False)):
        return False
    if "inside_2" in enabled and not bool(d2.get("Inside2", False)):
        return False
    if "above_sma" in enabled:
        sma_col = f"SMA{args.above_sma}"
        if pd.isna(d2.get(sma_col)) or not (d2["Close"] > d2[sma_col]):
            return False
    if "below_sma" in enabled:
        sma_col = f"SMA{args.below_sma}"
        if pd.isna(d2.get(sma_col)) or not (d2["Close"] < d2[sma_col]):
            return False
    if "trend_slope" in enabled:
        if pd.isna(d2.get("SMA20")) or pd.isna(d2.get("SMA20_5dago")):
            return False
        if (d2["SMA20"] - d2["SMA20_5dago"]) <= args.trend_slope:
            return False
    if "body_pct" in enabled and (
        pd.isna(d2.get("BodyPct")) or d2["BodyPct"] < args.body_pct_min
    ):
        return False
    if "upper_wick" in enabled and (
        pd.isna(d2.get("UpperWickPct")) or d2["UpperWickPct"] > args.upper_wick_max
    ):
        return False
    if "lower_wick" in enabled and (
        pd.isna(d2.get("LowerWickPct")) or d2["LowerWickPct"] > args.lower_wick_max
    ):
        return False
    if "pullback_pct" in enabled and (
        pd.isna(d2.get("PullbackPct20")) or d2["PullbackPct20"] > args.pullback_pct_max
    ):
        return False
    if "gap" in enabled:
        if pd.isna(d3.get("Close")) or pd.isna(d2.get("Open")) or d3["Close"] == 0:
            return False
        gap_pct = abs((d2["Open"] - d3["Close"]) / d3["Close"]) * 100.0
        if gap_pct < args.min_gap_pct:
            return False
    return True
