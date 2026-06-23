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
import streamlit.components.v1 as components
import yaml
import numpy as np
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

    # Initialize base seed for projection matrices (reproducibility)
    if "projection_seed" not in st.session_state:
        # Load from config, or generate random seed if not specified
        base_seed = CONFIG.get("projections", {}).get("projection_seed", None)
        if base_seed is None:
            # No seed in config - generate random seed for this session
            base_seed = np.random.randint(0, 2**31 - 1)
        st.session_state.projection_seed = base_seed

    # Create projection matrices with derived seeds (W_k, W_v, W_q)
    # Using base_seed, base_seed+1, base_seed+2 ensures different matrices
    base_seed = st.session_state.projection_seed

    if "W_k" not in st.session_state:
        st.session_state.W_k = ProjectionMatrix(
            input_dim=_EMBEDDING_DIM,
            output_dim=_DEFAULT_R,
            scale=_INIT_SCALE,
            seed=base_seed  # Key projection uses base seed
        )

    if "W_v" not in st.session_state:
        st.session_state.W_v = ProjectionMatrix(
            input_dim=_EMBEDDING_DIM,
            output_dim=_DEFAULT_R,
            scale=_INIT_SCALE,
            seed=base_seed + 1  # Value projection uses seed with base + 1
        )

    if "W_q" not in st.session_state:
        st.session_state.W_q = ProjectionMatrix(
            input_dim=_EMBEDDING_DIM,
            output_dim=_DEFAULT_R,
            scale=_INIT_SCALE,
            seed=base_seed + 2  # Query projection uses seed with base + 2
        )

    # Create SSW and inject the seeded projection matrices
    if "ssw" not in st.session_state:
        ssw = SequenceStateWrite()
        # Inject the globally cached model so the lazy-loader in
        # SentenceEncoderWrapper never triggers a fresh model load.
        ssw.encoder._encoder = _load_encoder_model()
        # Inject seeded projection matrices for reproducibility
        ssw.W_k = st.session_state.W_k
        ssw.W_v = st.session_state.W_v
        st.session_state.ssw = ssw

    # Create Registry and inject the seeded query projection
    if "registry" not in st.session_state:
        registry = SentenceRegistry()
        # Inject seeded query projection for reproducibility
        registry.W_q = st.session_state.W_q
        st.session_state.registry = registry

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

    if "current_input" not in st.session_state:
        st.session_state.current_input = ""

    if "last_insert_step" not in st.session_state:
        st.session_state.last_insert_step = None

    if "last_insert_count" not in st.session_state:
        st.session_state.last_insert_count = None


# ---------------------------------------------------------------------------
# Full session reset — rebuilds all stateful objects for a given r.
# Called both from the "Apply" path (empty state, r changed) and the
# two-step "Yes, reset everything" confirmation path (data exists, r changed
# or user triggered a plain reset without changing r).
# ---------------------------------------------------------------------------


def _apply_reset(new_r: int) -> None:
    """Rebuild all session objects for the given memory size and clear state.

    Creates fresh OSAMState, SequenceStateWrite, and SentenceRegistry
    instances sized to new_r. Projection matrices are recreated with the
    SAME base seeds regardless of dimension, ensuring consistency:
    - Same dimension → identical matrices (bit-for-bit)
    - Different dimension → same random initialization pattern
    - Reverting to previous dimension → recovers original matrices

    This aligns with paper's assumption of fixed, trained projections.

    Args:
        new_r: Target memory size (4 ≤ new_r ≤ 16).
    """
    # Fresh memory state
    st.session_state.state = OSAMState(r=new_r, beta=_DEFAULT_BETA)

    # Always use base seed - only dimension changes, not the seed
    # This ensures same dimension always produces identical matrices
    base_seed = st.session_state.projection_seed

    # Recreate projection matrices with base seeds (dimension-independent)
    st.session_state.W_k = ProjectionMatrix(
        input_dim=_EMBEDDING_DIM,
        output_dim=new_r,
        scale=_INIT_SCALE,
        seed=base_seed  # Always use base seed
    )
    st.session_state.W_v = ProjectionMatrix(
        input_dim=_EMBEDDING_DIM,
        output_dim=new_r,
        scale=_INIT_SCALE,
        seed=base_seed + 1  # Offset for different matrix
    )
    st.session_state.W_q = ProjectionMatrix(
        input_dim=_EMBEDDING_DIM,
        output_dim=new_r,
        scale=_INIT_SCALE,
        seed=base_seed + 2  # Offset for different matrix
    )

    # Create new SSW and inject the seeded projection matrices
    new_ssw = SequenceStateWrite()
    new_ssw.encoder._encoder = _load_encoder_model()
    new_ssw.W_k = st.session_state.W_k  # Inject seeded matrix
    new_ssw.W_v = st.session_state.W_v  # Inject seeded matrix
    st.session_state.ssw = new_ssw

    # Create new Registry and inject the seeded query projection
    new_registry = SentenceRegistry()
    new_registry.W_q = st.session_state.W_q  # Inject seeded matrix
    st.session_state.registry = new_registry

    # Clear all derived / transient state
    st.session_state.last_diff = None
    st.session_state.last_results = None
    st.session_state.last_query = None
    st.session_state.confirm_reset = False
    # Note: r_input widget key is NOT set here — after a successful apply/reset,
    # state.r == new_r and the widget already holds new_r, so no sync needed.


# ---------------------------------------------------------------------------
# Heatmap Component — Custom HTML/CSS visualization
# ---------------------------------------------------------------------------


def _value_to_hex(value: float, vmax: float) -> str:
    """
    Map a scalar value to RdBu diverging colorscale hex color.

    Uses a blue-white-red diverging scale where:
    - Negative values → shades of blue (#2166ac to #f7f7f7)
    - Zero → neutral white/gray (#f7f7f7)
    - Positive values → shades of red (#f7f7f7 to #b2182b)

    Args:
        value: Scalar value to map (will be clipped to [-vmax, +vmax])
        vmax: Maximum absolute value for normalization

    Returns:
        Hex color string (e.g., "#2166ac")
    """
    # Normalize to [-1, 1] range
    normed = np.clip(value / vmax, -1.0, 1.0)

    if normed < 0:
        # Blue side: interpolate from dark blue to white
        t = (normed + 1.0)  # maps [-1, 0] → [0, 1]
        # Dark blue RGB: (33, 102, 172) → White RGB: (247, 247, 247)
        r = int(33 + t * (247 - 33))
        g = int(102 + t * (247 - 102))
        b = int(172 + t * (247 - 172))
    else:
        # Red side: interpolate from white to dark red
        t = normed  # maps [0, 1] → [0, 1]
        # White RGB: (247, 247, 247) → Dark red RGB: (178, 24, 43)
        r = int(247 + t * (178 - 247))
        g = int(247 + t * (24 - 247))
        b = int(247 + t * (43 - 247))

    return f"#{r:02x}{g:02x}{b:02x}"


def _render_empty_state_table(r: int) -> str:
    """
    Generate HTML for empty state heatmap (all zeros).

    Args:
        r: Matrix dimension (r×r)

    Returns:
        HTML string with faint grid and empty state message
    """
    html = '<table class="heatmap empty">\n'
    for i in range(r):
        html += "  <tr>\n"
        for j in range(r):
            html += '    <td class="cell" style="background-color: #f9f9f9; color: #ccc;">0.000</td>\n'
        html += "  </tr>\n"
    html += "</table>\n"
    html += '<div class="empty-message">Memory is empty — insert a sentence to begin</div>\n'
    return html


def _render_color_legend(vmax: float) -> str:
    """
    Generate HTML for color legend bar with min/max labels.

    Args:
        vmax: Maximum absolute value in the matrix

    Returns:
        HTML string with gradient bar and labels
    """
    return f"""
<div class="legend-container">
    <div class="legend-bar"></div>
    <div class="legend-labels">
        <span>-{vmax:.3f}</span>
        <span>0.000</span>
        <span>+{vmax:.3f}</span>
    </div>
</div>
"""


def render_heatmap(
    S: np.ndarray,
    last_diff: np.ndarray | None,
    step_count: int,
    config: dict,
) -> None:
    """
    Render the memory matrix as a custom HTML/CSS heatmap.

    Features:
    - RdBu diverging colorscale (blue=negative, white=zero, red=positive)
    - Pulse animation: changed cells flash gold→final color over 400ms
    - Empty state message when matrix is all zeros
    - Step counter caption
    - Color legend with value range

    Args:
        S: Memory matrix (r×r numpy array)
        last_diff: Difference matrix from last update (for pulse animation)
        step_count: Number of sentences written (for caption)
        config: Configuration dict with visualization settings

    Side effects:
        - Renders HTML via st.components.v1.html()
        - Clears st.session_state.last_diff after rendering (prevents re-animation)
    """
    r = S.shape[0]
    vmax = max(abs(S).max(), 1e-6)  # Avoid division by zero
    is_empty = np.allclose(S, 0.0, atol=1e-9)
    
    # Get pulse animation settings from config
    threshold = config["visualization"]["diff_highlight_threshold"]  # 0.001
    pulse_ms = config["visualization"]["pulse_duration_ms"]  # 1200
     
    # Detect changed cells (where abs(diff) > threshold)
    changed = (
        np.abs(last_diff) > threshold
        if last_diff is not None
        else np.zeros_like(S, dtype=bool)
    )

    # CSS styles (with pulse animation)
    html = f"""
<style>
    table.heatmap {{
        border-collapse: collapse;
        margin: 20px auto;
        font-family: monospace;
    }}
    
    td.cell {{
        width: 60px;
        height: 60px;
        text-align: center;
        vertical-align: middle;
        border: 1px solid #ddd;
        font-size: 11px;
        font-weight: 500;
    }}
    
    /* Pulse animation for changed cells */
    /* Holds gold color for 40% of duration, then fades to final color */
    @keyframes pulse {{
        0% {{
            background-color: #ffd700;  /* gold highlight */
        }}
        40% {{
            background-color: #ffd700;  /* hold gold */
        }}
        100% {{
            background-color: var(--final-color);  /* fade to final */
        }}
    }}
    
    td.cell.changed {{
        animation: pulse {pulse_ms}ms ease-in-out forwards;
    }}
    
    table.heatmap.empty {{
        opacity: 0.5;
    }}
    
    .empty-message {{
        text-align: center;
        color: #999;
        margin-top: 20px;
        font-style: italic;
        font-size: 14px;
    }}
    
    .caption {{
        text-align: center;
        color: #666;
        margin-top: 15px;
        font-size: 13px;
        font-weight: 500;
    }}
    
    .legend-container {{
        margin-top: 30px;
        text-align: center;
    }}
    
    .legend-bar {{
        background: linear-gradient(to right, 
            #2166ac 0%, #67a9cf 25%, #f7f7f7 50%, #ef8a62 75%, #b2182b 100%);
        height: 20px;
        width: 320px;
        margin: 10px auto;
        border: 1px solid #ddd;
        border-radius: 3px;
    }}
    
    .legend-labels {{
        display: flex;
        justify-content: space-between;
        width: 320px;
        margin: 5px auto;
        font-size: 12px;
        color: #666;
        font-family: monospace;
    }}
</style>
"""

    # Build table
    if is_empty:
        html += _render_empty_state_table(r)
    else:
        html += '<table class="heatmap">\n'
        for i in range(r):
            html += "  <tr>\n"
            for j in range(r):
                value = S[i, j]
                color = _value_to_hex(value, vmax)
                
                # Apply "changed" class if this cell was updated
                cell_class = "cell changed" if changed[i, j] else "cell"

                # Use CSS custom property for final color (enables animation)
                html += f'    <td class="{cell_class}" style="--final-color: {color}; background-color: {color};">'
                html += f"{value:.3f}</td>\n"
            html += "  </tr>\n"
        html += "</table>\n"

    # Add caption
    sentence_plural = "sentence" if step_count == 1 else "sentences"
    html += f'<div class="caption">Step {step_count} — {step_count} {sentence_plural} written</div>\n'

    # Add color legend
    html += _render_color_legend(vmax)

    # Render with dynamic height (adjust based on r)
    # Base: table (60px/row) + margins (60px) + caption (40px) + legend (80px) = ~240px overhead
    height = (r * 60) + 240
    components.html(html, height=height, scrolling=False)

    # Clear diff to prevent re-animation on next rerun
    st.session_state.last_diff = None


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
    # Two-column layout: Input & Registry (35%) | Matrix Visualization (65%)
    # ------------------------------------------------------------------

    col_left, col_right = st.columns([35, 65], gap="large")

    # ---------------------------------------------------------------
    # Left column: Input & Sentence Registry
    # ---------------------------------------------------------------
    with col_left:

        # [TASK: input-area] — text input, buttons (validate on click)
        st.subheader("Input")

        # Text area for sentence entry
        user_input = st.text_area(
            label="Enter a sentence to Insert or Retrieve",
            value=st.session_state.current_input,
            height=100,
            max_chars=_MAX_INPUT_LEN,
            placeholder="Type a sentence here (e.g., 'The quick brown fox jumps over the lazy dog')",
            key="text_input_area",
            label_visibility="collapsed",
        )
        # Sync to session state
        st.session_state.current_input = user_input

        # Button row: Insert | Retrieve (always enabled, validate on click)
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button(
                "Insert into Memory",
                key="insert_btn",
                type="primary",
                use_container_width=True,
            ):
                # Validate input
                if len(user_input.strip()) == 0:
                    st.error("⚠️ Please enter some text before inserting.")
                elif len(user_input) > _MAX_INPUT_LEN:
                    st.error(
                        f"⚠️ Input too long: {len(user_input)}/{_MAX_INPUT_LEN} characters. "
                        f"Please shorten your text by {len(user_input) - _MAX_INPUT_LEN} characters."
                    )
                else:
                    # Check capacity before encoding (avoid wasted work)
                    if len(st.session_state.registry.entries) >= _MAX_SENTENCES:
                        st.warning(
                            f"⚠️ Memory is full ({_MAX_SENTENCES}/{_MAX_SENTENCES} sentences). "
                            "Reset memory to add more sentences."
                        )
                    else:
                        # Execute write with loading spinner
                        with st.spinner("Encoding and writing to memory..."):
                            new_state, metadata = st.session_state.ssw.execute(
                                user_input,
                                st.session_state.state,
                                beta=st.session_state.beta,
                            )

                        # Update memory state
                        st.session_state.state = new_state

                        # Store sentence in registry
                        st.session_state.registry.store(
                            text=metadata["text"],
                            embedding=metadata["embedding"],
                            key=metadata["key"],
                            value=metadata["value"],
                            step_index=metadata["step_index"],
                        )

                        # Save diff for heatmap pulse animation (Step 2.5)
                        st.session_state.last_diff = metadata["state_diff"]

                        # Clear stale retrieval results
                        st.session_state.last_results = None

                        # Clear input field for next entry
                        st.session_state.current_input = ""

                        # Set success flag for feedback on next render
                        st.session_state.last_insert_step = metadata["step_index"] + 1
                        st.session_state.last_insert_count = len(
                            st.session_state.registry.entries
                        )

                        # Rerun to show cleared input and success message
                        st.rerun()

        with btn_col2:
            if st.button(
                "Retrieve from Memory",
                key="retrieve_btn",
                use_container_width=True,
            ):
                # Validate input
                if len(user_input.strip()) == 0:
                    st.error("⚠️ Please enter a query before retrieving.")
                else:
                    # [TASK: retrieve-flow] — to be implemented in Step 2.7
                    st.warning("Retrieve flow not yet implemented (Step 2.7).")
        
        # Show success feedback after buttons (same location as error messages)
        if st.session_state.last_insert_step is not None:
            st.success(f"Sentence inserted")
            # Clear the flag so it doesn't show on every rerun
            st.session_state.last_insert_step = None
            st.session_state.last_insert_count = None

        st.write("---")

        # Sentence Registry - shows all stored sentences
        st.subheader("Sentence Registry")
        
        registry = st.session_state.registry
        num_entries = len(registry.entries)
        
        # Header with count
        st.caption(f"**{num_entries} / {_MAX_SENTENCES}** stored")
        
        if num_entries == 0:
            st.info("No sentences stored yet. Insert a sentence to begin building memory.")
        else:
            # Display entries in reverse order (most recent first)
            st.write("")  # Small spacing
            
            for entry in reversed(registry.entries):
                step_num = entry.step_index + 1  # 0-indexed internally, display as 1-indexed
                text = entry.text
                
                # Truncate long sentences with ellipsis
                display_text = text if len(text) <= 60 else text[:57] + "..."
                
                # Display with step number and text
                st.markdown(f"**#{step_num}** · {display_text}")

        # [TASK: retrieve-flow] — retrieval results area will go here

    # ---------------------------------------------------------------
    # Right column: Memory Matrix Visualization
    # ---------------------------------------------------------------
    with col_right:

        # Memory matrix heatmap visualization
        st.subheader("Memory Matrix")
        render_heatmap(
            S=st.session_state.state.S,
            last_diff=st.session_state.last_diff,
            step_count=st.session_state.ssw.step_counter,
            config=CONFIG,
        )



if __name__ == "__main__":
    main()
