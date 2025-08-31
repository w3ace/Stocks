from __future__ import annotations

import altair as alt
import pandas as pd


def equity_curve(trades: pd.DataFrame) -> alt.Chart:
    """Return an Altair line chart of cumulative equity.

    When ``exit_day`` is present, gains are aggregated by day and the
    resulting cumulative change is plotted by date.  Otherwise the
    cumulative equity per trade index is shown.
    """

    if trades.empty:
        return alt.Chart(pd.DataFrame({"x": [], "equity": []})).mark_line()

    df = trades.copy()

    if "exit_day" in df.columns:
        df["exit_day"] = pd.to_datetime(df["exit_day"])
        df = df.sort_values("exit_day")

        daily = df.groupby("exit_day")["gain_loss_pct"].sum()
        daily = daily.reindex(
            pd.date_range(daily.index.min(), daily.index.max()), fill_value=0
        )

        equity = (1 + daily / 100).cumprod()
        plot_df = equity.reset_index()
        plot_df.columns = ["exit_day", "equity"]
        return (
            alt.Chart(plot_df)
            .mark_line()
            .encode(x=alt.X("exit_day:T", title="Date"), y="equity")
        )

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
