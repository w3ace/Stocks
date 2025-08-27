from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pandas as pd

from fetch_stock import fetch_stock


@dataclass
class OpenRangeBreakTotals:
    """Summary statistics from an opening range analysis."""

    total_days: int = 0
    total_trades: int = 0
    total_profit: float = 0.0
    total_top_profit: float = 0.0
    closed_higher_than_open: int = 0
    broke_low_first: int = 0
    broke_low_then_high: int = 0
    broke_high_first: int = 0
    broke_high_then_low: int = 0
    or_high_before_low: int = 0
    or_low_before_high: int = 0
    low_before_high_close_up: int = 0
    high_before_low_close_up: int = 0
    high_before_low_map: dict[pd.Timestamp, bool] = field(default_factory=dict)
    trade_details: list[dict[str, float | str | pd.Timestamp]] = field(
        default_factory=list
    )


def fetch_intraday(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str = "5m"
) -> pd.DataFrame:
    """Fetch intraday data for ``ticker`` using :func:`fetch_stock`."""
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    start_est = start.tz_localize("US/Eastern") if start.tzinfo is None else start.tz_convert("US/Eastern")
    start_est = start_est.normalize() + pd.Timedelta(hours=9, minutes=30)
    start = start_est.tz_convert("UTC")

    end_est = end.tz_localize("US/Eastern") if end.tzinfo is None else end.tz_convert("US/Eastern")
    if end_est.time() == pd.Timestamp("00:00", tz="US/Eastern").time():
        end_est = end_est.normalize() + pd.Timedelta(hours=16)
    end = end_est.tz_convert("UTC")

    data = fetch_stock(ticker, start_date=start, end_date=end, interval=interval)
    if data is None or data.empty:
        return pd.DataFrame()

    if "Datetime" in data.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(data["Datetime"], errors="coerce"))
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(data.index, errors="coerce"))

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    data.index = idx
    if "Datetime" in data.columns:
        data = data.drop(columns=["Datetime"])

    data.sort_index(inplace=True)
    data = data.loc[(data.index >= start) & (data.index <= end + timedelta(days=1))]
    return data


def determine_gain_or_loss(
    rest_of_day: pd.DataFrame,
    buy_price: float,
    loss_pct: float,
    profit_pct: float,
) -> tuple[str, float, pd.Timestamp, float]:
    """Return the trade outcome, exit price, time and top price."""
    if rest_of_day.empty:
        return "close", buy_price, pd.NaT, buy_price

    stop_price = buy_price * (1 - loss_pct / 100)
    target_price = buy_price * (1 + profit_pct / 100)

    loss_hit = rest_of_day[rest_of_day["Low"] <= stop_price]
    profit_hit = rest_of_day[rest_of_day["High"] >= target_price]

    # close
    if loss_hit.empty and profit_hit.empty:
        exit_time = rest_of_day.index[-1]
        exit_price = float(rest_of_day.iloc[-1]["Close"])
        top_price = float(rest_of_day["High"].max())
        return "close", exit_price, exit_time, top_price

    first_loss_time = loss_hit.index[0] if not loss_hit.empty else None
    first_profit_time = profit_hit.index[0] if not profit_hit.empty else None

    if first_profit_time is not None and (
        first_loss_time is None or first_profit_time < first_loss_time
    ):
        exit_time = first_profit_time
        exit_price = target_price

        # Determine end time: either the stop loss or the close of day
        last_time = rest_of_day.index[-1]
        end_time = min(first_loss_time, last_time) if first_loss_time is not None else last_time

        subset = rest_of_day.loc[exit_time:end_time]
        top_price = float(subset["High"].max())
        return "profit", exit_price, exit_time, top_price

    if first_loss_time is not None:
        exit_time = first_loss_time
        exit_price = stop_price
        top_price = exit_price
        return "loss", exit_price, exit_time, top_price

    exit_time = rest_of_day.index[-1]
    exit_price = float(rest_of_day.iloc[-1]["Close"])
    top_price = float(rest_of_day.loc[exit_time:]["High"].max())
    return "close", exit_price, exit_time, top_price


def calculate_open_range_pct(
    df: pd.DataFrame, open_range_minutes: int = 30
) -> pd.Series:
    """Return a Series of opening range percentages indexed by date."""
    if df.empty:
        return pd.Series(dtype=float)

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)

    pct_values = {}
    open_end = (
        pd.Timestamp("09:30") + timedelta(minutes=open_range_minutes)
    ).strftime("%H:%M")
    for date, day_df in grouped:
        morning = day_df.between_time("09:30", open_end)
        if morning.empty:
            continue
        or_high = morning["High"].max()
        or_low = morning["Low"].min()
        open_price = morning.iloc[0]["Open"]
        pct_values[pd.to_datetime(date)] = (or_high - or_low) / open_price * 100

    return pd.Series(pct_values).sort_index()


def analyze_open_range(
    df: pd.DataFrame,
    open_range_minutes: int = 30,
    *,
    loss_pct: float = 0.35,
    profit_pct: float = 1.0,
    filter: str = "MO",
    filter_offset: float = 1.0,
    max_trades: int | None = None,
) -> OpenRangeBreakTotals:
    """Analyze opening range breaks for each trading day.

    ``filter`` may contain multiple space-separated filters which must all
    evaluate to true for a trade to be taken. Prefix a filter with ``!`` to
    invert its logic.
    """
    if df.empty:
        return OpenRangeBreakTotals()

    df = df.tz_convert("US/Eastern")
    grouped = df.groupby(df.index.date)

    totals = OpenRangeBreakTotals()

    open_end = (
        pd.Timestamp("09:30") + timedelta(minutes=open_range_minutes)
    ).strftime("%H:%M")
    after_settlement = (
        pd.Timestamp(open_end) # + timedelta(minutes=5)
    ).strftime("%H:%M")

    prev_close = None

    for date, day_df in grouped:
        morning = day_df.between_time("09:30", open_end)
        near_closing = day_df.between_time(open_end, after_settlement)
        rest_of_day = day_df.between_time(after_settlement, "16:00")
        if morning.empty:
            continue
        or_high = morning["High"].max()
        or_low = morning["Low"].min()
        or_high_time = morning["High"].idxmax()
        or_low_time = morning["Low"].idxmin()
        open_price = morning.iloc[0]["Open"]
        close_price = day_df.iloc[-1]["Close"]
        if near_closing.empty:
            prev_close = close_price
            continue
        after_or_price = near_closing.iloc[0]["Open"]
        after_or_time = near_closing.index[0]

        if close_price > open_price:
            totals.closed_higher_than_open += 1

        should_buy = True
        gap_up = gap_down = False
        if prev_close is not None:
            gap_up = open_price > prev_close * filter_offset
            gap_down = open_price < prev_close * filter_offset
        for token in str(filter).split():
            negated = token.startswith("!")
            name = token[1:] if negated else token
            check = False
            if name.upper() == "MO":
                check = after_or_price > open_price * filter_offset
            elif name.upper() == "OM":
                check = open_price > after_or_price * filter_offset
            elif name.upper() == "ORM":
                check = after_or_price  > or_high * 0.9965
            elif name.upper() == "MPC":
                check = (
                    prev_close is not None
                    and after_or_price > prev_close * filter_offset
                )
            elif name.upper() == "GU":
                check = gap_up
            elif name.upper() == "GD":
                check = gap_down
            else:
                continue
            if negated:
                check = not check
            should_buy = should_buy and check

        if should_buy:
            buy = after_or_price
            outcome, sell, sell_time, top_price = determine_gain_or_loss(
                rest_of_day,
                buy_price=buy,
                loss_pct=loss_pct,
                profit_pct=profit_pct,
            )
            profit = ((sell - buy) / buy) * 100
            top_profit_pct = ((top_price - buy) / buy) * 100
            minutes = (
                (sell_time - after_or_time).total_seconds() / 60
                if pd.notna(sell_time)
                else None
            )
            stop_price = buy * (1 - loss_pct / 100)
            target_price = buy * (1 + profit_pct / 100)
            totals.total_trades += 1
            totals.total_profit += profit
            totals.total_top_profit += top_profit_pct
            totals.trade_details.append(
                {
                    "date": pd.to_datetime(date),
                    "time": after_or_time,
                    "open": float(open_price), 
                    "close": float(close_price),
                    "or_low": float(or_low),
                    "or_high": float(or_high),
                    "buy_price": float(buy),
                    "stop_price": float(stop_price),
                    "profit_price": float(target_price),
                    "profit": float(profit),
                    "top_profit": float(top_profit_pct),
                    "result": outcome,
                    "buy_time": after_or_time,
                    "sell_time": sell_time,
                    "minutes": minutes,
                }
            )
        if or_high_time < or_low_time:
            totals.or_high_before_low += 1
            if close_price > open_price:
                totals.high_before_low_close_up += 1
        elif or_low_time < or_high_time:
            totals.or_low_before_high += 1
            if close_price > open_price:
                totals.low_before_high_close_up += 1
        after_open = day_df[day_df.index > morning.index[-1]]
        if after_open.empty:
            totals.total_days += 1
            continue

        high_cross_time = None
        low_cross_time = None
        for idx, row in after_open.iterrows():
            if low_cross_time is None and row["Low"] <= or_low:
                low_cross_time = idx
            if high_cross_time is None and row["High"] >= or_high:
                high_cross_time = idx
            if low_cross_time is not None and high_cross_time is not None:
                break

        totals.total_days += 1

        if high_cross_time is not None and (low_cross_time is None or high_cross_time < low_cross_time):
            totals.broke_high_first += 1
            after_high = after_open.loc[high_cross_time:]
            totals.high_before_low_map[pd.to_datetime(date)] = True
            if (after_high["Low"] <= or_low).any():
                totals.broke_high_then_low += 1
        else:
            if low_cross_time is not None:
                totals.broke_low_first += 1
                after_low = after_open.loc[low_cross_time:]
                if (after_low["High"] >= or_high).any():
                    totals.broke_low_then_high += 1
            totals.high_before_low_map[pd.to_datetime(date)] = False

        prev_close = close_price

    return totals


class OpenRangeAnalyzer:
    """Helper class for running opening range analysis on a ticker.

    Parameters mirror :func:`analyze_open_range`. ``filter`` may contain multiple
    space-separated filters with optional ``!`` prefix to invert.
    """

    def __init__(
        self,
        *,
        interval: str = "5m",
        open_range_minutes: int = 30,
        loss_pct: float = 0.35,
        profit_pct: float = 1.0,
        filter: str = "MO",
        filter_offset: float = 1.0,
        max_trades: int | None = None,
    ) -> None:
        self.interval = interval
        self.open_range_minutes = open_range_minutes
        self.loss_pct = loss_pct
        self.profit_pct = profit_pct
        self.filter = filter
        self.filter_offset = filter_offset
        self.max_trades = max_trades

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return fetch_intraday(ticker, start, end, interval=self.interval)

    def analyze(self, df: pd.DataFrame) -> OpenRangeBreakTotals:
        return analyze_open_range(
            df,
            open_range_minutes=self.open_range_minutes,
            loss_pct=self.loss_pct,
            profit_pct=self.profit_pct,
            filter=self.filter,
            filter_offset=self.filter_offset,
            max_trades=self.max_trades,
        )

    def analyze_ticker(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[OpenRangeBreakTotals, pd.Series]:
        df = self.fetch(ticker, start, end)
        or_pct = calculate_open_range_pct(df, open_range_minutes=self.open_range_minutes)
        results = self.analyze(df)
        return results, or_pct
