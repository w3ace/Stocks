from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def export_trades(df: pd.DataFrame, path: Optional[str]) -> None:
    """Export *df* to *path* as CSV if *path* is provided."""
    if path:
        Path(path).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )
        df.to_csv(path, index=False)
