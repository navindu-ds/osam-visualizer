"""
OSAM Memory Visualizer — Streamlit Application

Interactive web application for the Online State of Associative Memory (OSAM)
module from δ-mem (Lei et al., 2026, https://arxiv.org/abs/2605.12357).

Users can write sentences into the r×r memory matrix via the delta-rule update
and query it to observe memory dynamics in real time.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import yaml
from pathlib import Path

from src.memory_state import OSAMState
from src.writing_strategies import SequenceStateWrite
from src.retrieval import SentenceRegistry

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OSAM Memory Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Config loading — cached so the YAML is read once per process
# ---------------------------------------------------------------------------


@st.cache_data
def load_config() -> dict:
    """Load defaults.yaml once and cache the result for the process lifetime."""
    config_path = Path(__file__).parent / "config" / "defaults.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

# Convenience aliases used throughout the file
_MAX_SENTENCES: int = CONFIG["storage"]["max_sentences"]
_MAX_INPUT_LEN: int = CONFIG["storage"]["max_input_length"]
_DEFAULT_R: int = CONFIG["memory"]["size"]
_DEFAULT_BETA: float = CONFIG["memory"]["write_strength"]
_TOP_K: int = CONFIG["retrieval"]["top_k"]
_PULSE_MS: int = CONFIG["visualization"]["pulse_duration_ms"]
_DIFF_THRESHOLD: float = CONFIG["visualization"]["diff_highlight_threshold"]
_MODEL_NAME: str = CONFIG["encoder"]["model_name"]

# ---------------------------------------------------------------------------
# Encoder caching — loads the SentenceTransformer model once per process.
# @st.cache_resource is process-level: all user sessions reuse the same model
# object, avoiding repeated ~80 MB loads. The spinner message is shown in the
# browser on the very first request that triggers the load.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading encoder model (one-time setup)...")
def _load_encoder_model():
    """Return a loaded SentenceTransformer model, cached for the process lifetime."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


# ---------------------------------------------------------------------------
# Session state initialisation — called once per user session.
# Creates isolated OSAMState, SequenceStateWrite, and SentenceRegistry objects
# per session, then patches the encoder wrapper to use the cached model so no
# second model load ever occurs.
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialise all per-session state keys if they do not yet exist.

    Safe to call on every rerun — only populates keys that are absent,
    leaving any existing state (written sentences, retrieval results, etc.)
    completely untouched.
    """
    if "state" not in st.session_state:
        st.session_state.state = OSAMState(r=_DEFAULT_R, beta=_DEFAULT_BETA)

    if "ssw" not in st.session_state:
        ssw = SequenceStateWrite()
        # Inject the globally cached model so the lazy-loader in
        # SentenceEncoderWrapper never triggers a fresh model load.
        ssw.encoder._encoder = _load_encoder_model()
        st.session_state.ssw = ssw

    if "registry" not in st.session_state:
        st.session_state.registry = SentenceRegistry()

    if "last_diff" not in st.session_state:
        st.session_state.last_diff = None

    if "last_results" not in st.session_state:
        st.session_state.last_results = None

    if "last_query" not in st.session_state:
        st.session_state.last_query = None

    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    if "beta" not in st.session_state:
        st.session_state.beta = _DEFAULT_BETA


# ---------------------------------------------------------------------------
# [TASK: heatmap-component]
# render_heatmap(S, last_diff, step_count, config) → st.components.v1.html
# — to be implemented in task 5
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# [TASK: retrieve-flow]
# render_results(results, last_query) — ranked list with score bars
# — to be implemented in task 7
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# [TASK: side-panel]
# render_registry(registry, last_result_step_indices) — scrollable HTML panel
# — to be implemented in task 8
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the full OSAM Memory Visualizer UI."""

    init_session_state()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    st.title("OSAM Memory Visualizer")
    st.caption(
        "An interactive testbed for the Online State of Associative Memory module "
        "— δ-mem (Lei et al., 2026)"
    )

    # [TASK: header-config-expander] — parameters expander goes here

    st.divider()

    # ------------------------------------------------------------------
    # Two-column layout: main panel (70 %) | side panel (30 %)
    # ------------------------------------------------------------------

    col_main, col_side = st.columns([7, 3], gap="large")

    # ---------------------------------------------------------------
    # Main panel
    # ---------------------------------------------------------------
    with col_main:

        # [TASK: input-area] — text input, char counter, buttons

        st.write("---")  # temporary separator; removed once input area is wired

        # [TASK: heatmap-component] — memory matrix heatmap

        # [TASK: retrieve-flow] — retrieval results area

        st.write("---")  # temporary separator; removed once reset is wired

        # [TASK: reset-flow] — reset control with two-step confirmation

    # ---------------------------------------------------------------
    # Side panel
    # ---------------------------------------------------------------
    with col_side:

        # [TASK: side-panel] — sentence registry list
        st.subheader("Sentence Registry")
        st.caption("Stored sentences will appear here after your first Insert.")


if __name__ == "__main__":
    main()
