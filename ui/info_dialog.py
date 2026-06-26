"""
Info dialog for the OSAM Memory Visualizer.

Renders a scrollable st.dialog modal with guide and technical tabs,
pipeline diagram, LaTeX equations, and static asset references.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from ui.header_badges import render_header_badges

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ASSETS_DIR = _REPO_ROOT / "docs" / "assets" / "info"
_INFO_DIR = _REPO_ROOT / "docs" / "info"

_PAPER_URL = "https://arxiv.org/abs/2605.12357"
_GITHUB_URL = "https://github.com/navindu-ds/osam-visualizer"

_PIPELINE_SVG = "Pipeline.svg"
_PIPELINE_PNG = "Pipeline.png"
_PIPELINE_CAPTION = (
    "Write path (bottom) and read path (top) through the OSAM memory matrix."
)

_PAPER_VS_DEMO_ROWS: list[tuple[str, str, str]] = [
    ("Hidden state `x_t`", "LLM layer output", "SentenceTransformer embedding"),
    ("`W_k, W_v, W_q`", "Trained", "Fixed random init (seed 42)"),
    (
        "Write gate `β`",
        "Per-row vector `sigmoid(W_β x + b)`",
        "Scalar UI slider (same β for all rows)",
    ),
    ("Delta update + read", "Exact OSAM", "**Faithful**"),
    (
        "Output pathway",
        "Low-rank attention correction → LLM text",
        "Registry cosine lookup (demo-only)",
    ),
    ("Writing strategies", "TSW / SSW / MSW", "**SSW only** (one update per sentence)"),
]

_HEATMAP_SAMPLES: list[tuple[str, str]] = [
    (
        "heatmap_live_state.png",
        "Live memory state `S` after several writes — consolidated matrix with fixed color legend.",
    ),
    (
        "heatmap_net_change.png",
        "Net Change ($S_t - S_{t-1}$) for one selected sentence — cell-wise update at that step.",
    ),
    (
        "heatmap_components.png",
        "Component view (e.g. Write New Value) with β caption and shared color scale.",
    ),
]


def _load_markdown(relative_path: str) -> str:
    path = _INFO_DIR / relative_path
    return path.read_text(encoding="utf-8")


def _render_svg(filename: str, caption: str | None = None) -> None:
    """Embed SVG as a base64 image (inline SVG markup is stripped by Streamlit)."""
    path = _ASSETS_DIR / filename
    if not path.is_file():
        st.info(f"Diagram `{filename}` not found.")
        return
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{b64}" alt="{filename}" '
        f'style="width:100%;max-width:100%;height:auto;" />',
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def _render_image(filename: str, caption: str) -> None:
    path = _ASSETS_DIR / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(
            f"Screenshot `{filename}` not found. "
            f"See `docs/assets/info/README.md` for capture instructions."
        )


def _render_pipeline_diagram() -> None:
    """Show pipeline diagram: SVG preferred, PNG fallback."""
    if (_ASSETS_DIR / _PIPELINE_SVG).is_file():
        _render_svg(_PIPELINE_SVG, caption=_PIPELINE_CAPTION)
    elif (_ASSETS_DIR / _PIPELINE_PNG).is_file():
        _render_image(_PIPELINE_PNG, caption=_PIPELINE_CAPTION)
    else:
        st.info(
            f"Pipeline diagram not found (`{_PIPELINE_SVG}` or `{_PIPELINE_PNG}`)."
        )


def _render_shared_header() -> None:
    st.markdown(
        "Interactive testbed for a simplified version of the **Online State of Associative Memory (OSAM)** "
        f"from [δ-mem (Lei et al., 2026)]({_PAPER_URL})."
    )

    render_header_badges(_PAPER_URL, _GITHUB_URL)

    st.divider()
    st.markdown("### Pipeline")
    _render_pipeline_diagram()

    st.markdown("### Quick start")
    st.markdown(
        """
1. **Insert** a sentence — watch the memory matrix update.
2. **Repeat** with a few more sentences.
3. **Retrieve** with a question — see ranked similarity scores (read-only).
4. **Reset** to clear memory and start over.
        """
    )


def _render_guide_tab() -> None:
    st.markdown(_load_markdown("guide.md"))

    st.divider()
    st.markdown("### Sample heatmaps")
    for filename, caption in _HEATMAP_SAMPLES:
        _render_image(filename, caption)


def _render_technical_tab() -> None:
    st.markdown(_load_markdown("technical.md"))

    st.divider()
    st.markdown("### Write update")
    st.latex(r"S_t = (1-\beta)\, S_{t-1} + \beta\, (v_t - S_{t-1} k_t)\, k_t^\top")

    st.markdown("### State formation (three components)")
    st.latex(
        r"S_t = \underbrace{(1-\beta) S_{t-1}}_{\text{Retention}}"
        r"+ \underbrace{-\beta (S_{t-1} k_t)\, k_t^\top}_{\text{Erase prediction}}"
        r"+ \underbrace{\beta v_t k_t^\top}_{\text{Write new value}}"
    )

    _render_svg(
        "component_decomposition.svg",
        caption="Additive components that form S_t (not net change).",
    )

    st.markdown("### Read")
    st.latex(r"r_t = S\, q_t")
    st.markdown("Demo scoring: `score(i) = cosine(r_t, v_i)` over registry value vectors.")

    st.divider()
    st.markdown("### Paper vs this demo")
    st.table(
        {
            "Step": [row[0] for row in _PAPER_VS_DEMO_ROWS],
            "Paper": [row[1] for row in _PAPER_VS_DEMO_ROWS],
            "This demo": [row[2] for row in _PAPER_VS_DEMO_ROWS],
        }
    )


@st.dialog("How OSAM Memory Works", width="large")
def render_info_dialog() -> None:
    """Open the info modal with guide and technical content."""
    _render_shared_header()

    guide_tab, technical_tab = st.tabs(["Guide", "Technical"])
    with guide_tab:
        _render_guide_tab()
    with technical_tab:
        _render_technical_tab()
