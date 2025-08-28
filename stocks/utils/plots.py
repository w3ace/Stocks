from __future__ import annotations

import altair as alt
import pandas as pd


def equity_curve(trades: pd.DataFrame) -> alt.Chart:
    """Return an Altair line chart of cumulative equity."""
    if trades.empty:
        return alt.Chart(pd.DataFrame({"x": [], "equity": []})).mark_line()
    df = trades.copy()
    df["equity"] = (1 + df["gain_loss_pct"] / 100).cumprod()
    df["trade"] = range(len(df))
    return alt.Chart(df).mark_line().encode(x="trade", y="equity")


def gain_loss_bar(trades: pd.DataFrame) -> alt.Chart:
    """Return an Altair bar chart of gain/loss percentages by exit day."""
    if trades.empty:
        return alt.Chart(
            pd.DataFrame({"x": [], "gain_loss_pct": []})
        ).mark_bar()
    df = trades.copy()
    df["exit_day"] = df["exit_day"].astype(str)
    return alt.Chart(df).mark_bar().encode(
        x="exit_day", y="gain_loss_pct"
    )
