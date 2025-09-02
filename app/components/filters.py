import streamlit as st

PATTERNS = ["4OUED", "3OUED", "4DUED", "3DUED", "4ODEU", "3ODEU"]


def pattern_selector() -> str:
    """Return a user supplied pattern string.

    Previously this component offered a fixed set of options.  The Taygetus
    backtester now supports a richer pattern syntax so we expose a free-form
    text input while keeping the function name for backwards compatibility.
    """

    return st.text_input("Pattern", PATTERNS[0])
