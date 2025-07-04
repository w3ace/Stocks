# Stocks Dayparts
Analyze and graph dayparts of stocks looking for trends.

The analysis scripts now include a data point for the overnight gap
between the previous day's close (4 PM) and the next day's open at
9:30 AM.  This is reported as `PrevClose_Open_Gap` (absolute) and
`PrevClose_Open_Gap_Pct` (percentage).

The opening range (the first hour high/low difference) is reported as
`Open_Range_Pct` relative to the day's open with a running total
`Cumulative_Open_Range_Pct`.

## Portfolios

`backtest.py` supports saved portfolios. Prefix a portfolio name
with `+` to load tickers from a file in the `portfolios` directory. For
example, running:

```
python backtest.py +M9 --period 6mo
```

reads tickers from `portfolios/M9.txt` which might contain:

```
AMZN MSFT AAPL NVDA META GOOG TSLA MU AVGO
```

Duplicate tickers are ignored, so the portfolio and command-line
arguments can safely include overlapping symbols.

If you omit both `--period` and `--start`, `backtest.py` will
analyze a single day automatically. When run before 9:30 AM US/Eastern it
uses the previous trading day (adjusting for weekends); otherwise it uses
the current day.

The script also provides `--profit-pct` and `--loss-pct` options to
control the intraday profit target and stop loss percentages when a trade
is taken after the opening range.
Use `--filter` to control the relationship between the opening range
close (Mark) and the day's open that must be satisfied before entering a
trade. `--filter-offset` multiplies the open price used in that check.
Available filters include `MO` (Mark > Open), `OM` (Open > Mark), `ORM`
(Buy Price * 1.002 > Open Range High), `GU` (Open above previous close)
and `GD` (Open below previous close).

When the analysis completes, all trades are written to `./output/<timestamp>_trades.csv` and a per-ticker summary is saved to `./output/<timestamp>_tickers.csv`. The summary lists the total number of trades, the percentage of profitable trades, and the cumulative profit for each ticker. It also includes `total_top_profit`, the sum of potential profits based on each trade's peak price.
The trades file includes a `profit_or_loss` column after `sell_time` showing whether each trade hit the profit target, stop loss, or closed at the end of the day.
Each trade also reports `top_profit`, the percent gain from entry to the highest price reached before exiting.
Pass `--output-trades` to print each trade in the terminal. Use `--tickers` or
`--output-tickers` to display the per-ticker summary in an ASCII table after the trades.
The summary table is ordered by `total_profit` descending and entries
with profits less than the value passed to `--min-profit` are omitted.
Pass `--plot daily` to generate a single plot summarizing daily activity. The
profit line appears on the left y-axis and a stacked bar chart on the right
shows the number of profit, close, and loss trades for each day. The plot also
includes a dashed line depicting the average top profit.
