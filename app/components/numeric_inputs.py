from decimal import Decimal, ROUND_HALF_UP

import streamlit as st


def _quantize(value: float, step: float) -> float:
    q = (
        Decimal(str(value)) / Decimal(str(step))
    ).quantize(0, rounding=ROUND_HALF_UP)
    return float(q * Decimal(str(step)))


def slider_with_input(
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float,
    key: str,
) -> float:
    """Return a value from a slider synced with a number input and nudge
    buttons."""

    if key not in st.session_state:
        st.session_state[key] = _quantize(value, step)

    def _set_from_slider() -> None:
        st.session_state[key] = _quantize(
            st.session_state[f"{key}_slider"], step
        )

    def _set_from_input() -> None:
        st.session_state[key] = _quantize(
            st.session_state[f"{key}_input"], step
        )

    col_a, col_b, col_c = st.columns([3, 1, 2], vertical_alignment="center")
    with col_a:
        st.slider(
            label,
            min_value,
            max_value,
            st.session_state[key],
            step=step,
            key=f"{key}_slider",
            on_change=_set_from_slider,
        )

    with col_b:
        btn_col1, btn_col2 = st.columns(2)
        btn_col1.button(
            "−",
            use_container_width=True,
            on_click=lambda: st.session_state.update(
                {key: _quantize(st.session_state[key] - step, step)}
            ),
        )
        btn_col2.button(
            "+",
            use_container_width=True,
            on_click=lambda: st.session_state.update(
                {key: _quantize(st.session_state[key] + step, step)}
            ),
        )

    with col_c:
        st.number_input(
            "Exact",
            min_value=min_value,
            max_value=max_value,
            step=step,
            value=st.session_state[key],
            key=f"{key}_input",
            on_change=_set_from_input,
            format="%.4f" if step < 1 else "%.0f",
        )

    return st.session_state[key]
