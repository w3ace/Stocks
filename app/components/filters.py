import streamlit as st

PATTERNS = ["3E", "3EC", "3D", "3DC", "3EU", "4EU"]


def pattern_selector() -> str:
    return st.selectbox("Pattern", PATTERNS)
