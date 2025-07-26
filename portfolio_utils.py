from pathlib import Path
from typing import Iterable, Sequence
import re


def expand_ticker_args(ticker_args: Iterable[str]) -> list[str]:
    """Expand portfolio names prefixed with ``+`` or ``-`` to tickers.

    Tokens beginning with ``+`` are treated as filenames in the
    ``portfolios`` directory.  Their tickers are added to the returned
    list.  Tokens beginning with ``-`` also reference files under
    ``portfolios`` but the tickers they contain are removed from the
    final result.  Duplicate tickers are removed while preserving order
    after all additions and exclusions are processed.
    """

    expanded: list[str] = []
    exclude_files: list[str] = []
    for token in ticker_args:
        if token.startswith("+"):
            name = token[1:]
            path = Path("portfolios") / name
            if path.exists():
                expanded.extend(path.read_text().split())
            else:
                print(f"Portfolio file not found: {path}")
        elif token.startswith("-"):
            exclude_files.append(token[1:])
        else:
            expanded.append(token)

    # Remove duplicates while preserving order
    result = list(dict.fromkeys(expanded))

    # Exclude tickers from ``-`` portfolios
    for name in exclude_files:
        path = Path("portfolios") / name
        if path.exists():
            exclusions = set(path.read_text().split())
            result = [t for t in result if t not in exclusions]
        else:
            print(f"Portfolio file not found: {path}")

    return result


def sanitize_ticker_string(tickers: Sequence[str] | str) -> str:
    """Return combined *tickers* string stripped of spaces and special characters."""

    if isinstance(tickers, str):
        combined = tickers
    else:
        combined = "".join(tickers)
    return re.sub(r"[^A-Za-z0-9]", "", combined)
