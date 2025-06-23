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

`open_range_break.py` supports saved portfolios. Prefix a portfolio name
with `+` to load tickers from a file in the `portfolios` directory. For
example, running:

```
python open_range_break.py +M9 --period 6mo
```

reads tickers from `portfolios/M9.txt` which might contain:

```
AMZN MSFT AAPL NVDA META GOOG TSLA MU AVGO
```

If you omit both `--period` and `--start`, `open_range_break.py` will
analyze a single day automatically. When run before 9:30 AM US/Eastern it
uses the previous trading day; otherwise it uses the current day.
