from pathlib import Path
from typing import Iterable


def expand_ticker_args(ticker_args: Iterable[str]) -> list[str]:
    """Expand portfolio names prefixed with ``+`` to tickers.

    Tokens beginning with ``+`` are treated as filenames in the
    ``portfolios`` directory. The tickers inside are added to the
    returned list. Duplicate tickers are removed while preserving order.
    """
    expanded: list[str] = []
    for token in ticker_args:
        if token.startswith("+"):
            name = token[1:]
            path = Path("portfolios") / name
            if path.exists():
                expanded.extend(path.read_text().split())
            else:
                print(f"Portfolio file not found: {path}")
        else:
            expanded.append(token)
    # Remove duplicates while preserving order
    return list(dict.fromkeys(expanded))
