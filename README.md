# Stocks

Hybrid Python project providing a reusable library, command line tools and a
Streamlit app for simple pattern backtests.

## Quickstart

```bash
pip install -r requirements.txt
python -m cli.taygetus_backtest --ticker AAPL --pattern 3E --period 1y
python -m cli.eldorado_backtest --ticker AAPL --period 1y
make run-app  # launches Streamlit on http://localhost:8501
```

## Development

```bash
make lint
make test
```

## Layout

```
stocks/           # core library
cli/              # command line interfaces
app/              # Streamlit application
```

## Future work

- authentication and user storage
- multi-ticker backtests
- optional CSV export and persistence
