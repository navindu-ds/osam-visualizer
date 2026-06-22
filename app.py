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
from src.encoder import ProjectionMatrix

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
_EMBEDDING_DIM: int = CONFIG["projections"]["input_dim"]
_INIT_SCALE: float = CONFIG["projections"]["init_scale"]

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

    if "top_k" not in st.session_state:
        st.session_state.top_k = _TOP_K


# ---------------------------------------------------------------------------
# Full session reset — rebuilds all stateful objects for a given r.
# Called both from the "Apply" path (empty state, r changed) and the
# two-step "Yes, reset everything" confirmation path (data exists, r changed
# or user triggered a plain reset without changing r).
# ---------------------------------------------------------------------------


def _apply_reset(new_r: int) -> None:
    """Rebuild all session objects for the given memory size and clear state.

    Creates fresh OSAMState, SequenceStateWrite, and SentenceRegistry
    instances sized to new_r.  If new_r differs from the config default
    the projection matrices are replaced in-place after construction so
    that W_k, W_v (in SSW) and W_q (in registry) all have shape
    (new_r, _EMBEDDING_DIM), matching the new state matrix dimension.

    Args:
        new_r: Target memory size (4 ≤ new_r ≤ 16).
    """
    # Fresh memory state
    st.session_state.state = OSAMState(r=new_r, beta=_DEFAULT_BETA)

    # Fresh SSW — built from config (output_dim = _DEFAULT_R = 8).
    # Replace projection matrices when new_r != config default so that
    # all matrix shapes stay consistent with the new state dimension.
    new_ssw = SequenceStateWrite()
    new_ssw.encoder._encoder = _load_encoder_model()
    if new_r != _DEFAULT_R:
        new_ssw.W_k = ProjectionMatrix(
            input_dim=_EMBEDDING_DIM, output_dim=new_r, scale=_INIT_SCALE
        )
        new_ssw.W_v = ProjectionMatrix(
            input_dim=_EMBEDDING_DIM, output_dim=new_r, scale=_INIT_SCALE
        )
    st.session_state.ssw = new_ssw

    # Fresh registry — same post-construction patch pattern for W_q.
    new_registry = SentenceRegistry()
    if new_r != _DEFAULT_R:
        new_registry.W_q = ProjectionMatrix(
            input_dim=_EMBEDDING_DIM, output_dim=new_r, scale=_INIT_SCALE
        )
    st.session_state.registry = new_registry

    # Clear all derived / transient state
    st.session_state.last_diff = None
    st.session_state.last_results = None
    st.session_state.last_query = None
    st.session_state.confirm_reset = False
    # Note: r_input widget key is NOT set here — after a successful apply/reset,
    # state.r == new_r and the widget already holds new_r, so no sync needed.


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

    # Apply any queued widget-value resets BEFORE the widgets render.
    # Streamlit forbids setting a widget's key after it has been instantiated
    # in the current run, so Cancel handlers queue via _reset_r_to instead.
    if "_reset_r_to" in st.session_state:
        st.session_state["r_input"] = st.session_state.pop("_reset_r_to")

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    st.title("OSAM Memory Visualizer")
    st.caption(
        "An interactive testbed for the Online State of Associative Memory module "
        "— δ-mem (Lei et al., 2026)"
    )

    # ------------------------------------------------------------------
    # Parameters — compact single row.
    # r (architectural, reset-required) | β (runtime) | top_k (runtime)
    # ------------------------------------------------------------------

    pc1, _sp1, pc2, _sp2, pc3 = st.columns([2, 1, 3, 1, 2])

    with pc1:
        new_r = st.number_input(
            "Memory size r",
            min_value=4,
            max_value=16,
            step=1,
            value=st.session_state.state.r,
            key="r_input",
            help=(
                "Dimension of the r×r state matrix S (range 4–16, default 8). "
                "Changing this requires a full reset: W_k, W_v, W_q are rebuilt "
                "to match the new output dimension."
            ),
        )

    with pc2:
        new_beta = st.slider(
            "Write strength β",
            min_value=0.01,
            max_value=0.50,
            value=float(st.session_state.beta),
            step=0.01,
            format="%.2f",
            help=(
                "How strongly each new sentence overwrites existing memory. "
                "Higher = more aggressive write; lower = more retention. "
                "Takes effect on the next Insert — no reset required."
            ),
        )
        if new_beta != st.session_state.beta:
            st.session_state.beta = new_beta

    with pc3:
        new_top_k = st.number_input(
            "Top-k results",
            min_value=1,
            max_value=10,
            step=1,
            value=st.session_state.top_k,
            key="top_k_input",
            help=(
                "Number of ranked results to return per Retrieve query. "
                "Takes effect on the next Retrieve — no reset required."
            ),
        )
        if int(new_top_k) != st.session_state.top_k:
            st.session_state.top_k = int(new_top_k)

    # Status + fixed-config reference on one compact caption line
    _step_count = st.session_state.ssw.step_counter
    _reg_len = len(st.session_state.registry)
    st.caption(
        f"**{st.session_state.state.r}×{st.session_state.state.r}** matrix  ·  "
        f"**{_reg_len}/{_MAX_SENTENCES}** sentences  ·  "
        f"**{_step_count}** write step{'s' if _step_count != 1 else ''}  ·  "
        f"max input {_MAX_INPUT_LEN} chars"
    )

    # r change guard — inline directly below the status line
    if int(new_r) != st.session_state.state.r:
        if st.session_state.ssw.step_counter == 0:
            # Nothing written yet — single-step apply
            st.info(
                f"Memory size will change to **r = {int(new_r)}** "
                f"({int(new_r)}×{int(new_r)} matrix). No data will be lost."
            )
            if st.button("Apply new memory size", key="apply_r_btn"):
                _apply_reset(int(new_r))
                st.rerun()
        elif not st.session_state.confirm_reset:
            # Data exists — step 1: warn + offer reset
            st.warning(
                f"Changing to **r = {int(new_r)}** requires a full memory reset — "
                f"all {_step_count} written sentence(s) will be lost."
            )
            if st.button("Reset memory", key="reset_r_btn", type="primary"):
                st.session_state.confirm_reset = True
        else:
            # Step 2: explicit confirmation
            st.error(
                "**Confirm reset** — this will permanently clear the memory matrix "
                "and the sentence registry. This cannot be undone."
            )
            c_yes, c_no = st.columns(2)
            if c_yes.button("Yes, reset everything", key="confirm_yes_btn", type="primary"):
                _apply_reset(int(new_r))
                st.rerun()
            if c_no.button("Cancel", key="confirm_no_btn"):
                st.session_state.confirm_reset = False
                # Queue the widget reset for the next rerun — Streamlit forbids
                # writing a widget key after it has already been instantiated
                # in the current run, so we use a flag picked up before the
                # widget renders on the next pass.
                st.session_state["_reset_r_to"] = st.session_state.state.r
                st.rerun()
    else:
        if st.session_state.confirm_reset:
            st.session_state.confirm_reset = False

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

    # ---------------------------------------------------------------
    # Side panel
    # ---------------------------------------------------------------
    with col_side:

        # [TASK: side-panel] — sentence registry list
        st.subheader("Sentence Registry")
        st.caption("Stored sentences will appear here after your first Insert.")


if __name__ == "__main__":
    main()
