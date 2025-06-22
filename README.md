# Stocks Dayparts
Analyze and graph dayparts of stocks looking for trends.

The analysis scripts now include a data point for the overnight gap
between the previous day's close (4 PM) and the next day's open at
9:30 AM.  This is reported as `PrevClose_Open_Gap` (absolute) and
`PrevClose_Open_Gap_Pct` (percentage).

The opening range (the first hour high/low difference) is reported as
`Open_Range_Pct` relative to the day's open with a running total
`Cumulative_Open_Range_Pct`.
