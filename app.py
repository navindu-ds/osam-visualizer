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
_PAPER_URL: str = "https://arxiv.org/abs/2605.12357"
_GITHUB_URL: str = "https://github.com/navindu-ds/osam-visualizer"

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

    if "confirm_manual_reset" not in st.session_state:
        st.session_state.confirm_manual_reset = False

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

    if "write_history" not in st.session_state:
        st.session_state.write_history = []

    if "selected_step" not in st.session_state:
        st.session_state.selected_step = None

    if "selected_component_view" not in st.session_state:
        st.session_state.selected_component_view = "Net Change"


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
    st.session_state.write_history = []
    st.session_state.selected_step = None
    st.session_state.selected_component_view = "Net Change"
    if "global_component_view" in st.session_state:
        del st.session_state.global_component_view
    # Note: r_input widget key is NOT set here — after a successful apply/reset,
    # state.r == new_r and the widget already holds new_r, so no sync needed.


# ---------------------------------------------------------------------------
# Retrieval Results Display — Helper Functions
# ---------------------------------------------------------------------------


def _score_to_bar(score: float, width: int = 10) -> str:
    """
    Convert cosine similarity score to visual bidirectional bar using Unicode blocks.
    
    Cosine similarity ranges from -1 (anti-parallel) to +1 (parallel).
    Bar is center-anchored: negative scores extend left, positive extend right.
    
    Args:
        score: Cosine similarity score in [-1, 1]
        width: Number of blocks in the bar (default 10, split evenly left/right)
    
    Returns:
        String with center marker | and filled (█) and empty (░) Unicode blocks
    
    Examples:
        score = -1.0 → "█████|░░░░░" (full negative)
        score =  0.0 → "░░░░░|░░░░░" (neutral)
        score = +1.0 → "░░░░░|█████" (full positive)
    """
    # Clamp score to valid range
    score = max(-1.0, min(1.0, score))
    
    # Half-width for each side
    half_width = width // 2
    
    if score < 0:
        # Negative score: fill blocks on the left side
        magnitude = abs(score)  # 0 to 1
        filled_left = int(magnitude * half_width)
        empty_left = half_width - filled_left
        # Left side: empty then filled (reading right to left from center)
        left_side = "░" * empty_left + "█" * filled_left
        right_side = "░" * half_width
    else:
        # Positive score: fill blocks on the right side
        magnitude = score  # 0 to 1
        filled_right = int(magnitude * half_width)
        empty_right = half_width - filled_right
        left_side = "░" * half_width
        # Right side: filled then empty (reading left to right from center)
        right_side = "█" * filled_right + "░" * empty_right
    
    return left_side + "|" + right_side


def _render_result_item(rank: int, result: dict, is_top: bool = False) -> None:
    """
    Render a single retrieval result with rank, text, score, and visual bar.
    
    Args:
        rank: Result rank (1-indexed)
        result: Result dict with keys: text, score, step_index
        is_top: Whether this is the top result (for emphasis)
    """
    text = result["text"]
    score = result["score"]
    step_index = result["step_index"]
    
    # Create container for the result
    if is_top:
        # Top result gets special emphasis with info container
        with st.container():
            st.markdown(f"**🥇 #{rank}** · Sentence #{step_index + 1}")
            st.info(f"**{text}**")
            st.caption(f"Score: **{score:.2f}** {_score_to_bar(score)}")
    else:
        # Regular results
        with st.container():
            st.markdown(f"**#{rank}** · Sentence #{step_index + 1}")
            st.markdown(f"{text}")
            st.caption(f"Score: {score:.2f} {_score_to_bar(score)}")
    
    # Small spacing between results
    st.write("")


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
    <div class="scale-note">Color scale fixed at ±{vmax:.3f} (max |value| across all steps and components)</div>
</div>
"""


def _compute_global_vmax(write_history: list, current_S: np.ndarray) -> float:
    """
    Compute a shared color-scale maximum across all memory views.

    Uses the largest absolute cell value from the live matrix, every stored
    pre/post state, diff, and decomposition component so component toggles and
    sentence selections share one meaningful diverging scale.
    """
    vmax = float(np.max(np.abs(current_S)))
    for entry in write_history:
        S_before = entry["S_before"]
        diff = entry["diff"]
        k = entry["k"]
        v = entry["v"]
        beta = entry["beta"]

        S_t = S_before + diff
        retention = (1 - beta) * S_before
        erase = -beta * np.outer(S_before @ k, k)
        write = beta * np.outer(v, k)

        for arr in (S_before, S_t, diff, retention, erase, write):
            vmax = max(vmax, float(np.max(np.abs(arr))))

    return max(vmax, 1e-6)


def _resolve_component_view(requested: str, options: list[str]) -> str:
    """Map a persisted component label to a valid option for the current sentence."""
    if requested in options:
        return requested
    unavailable_fallbacks = {
        "Retention": "Net Change",
        "Erase Prediction": "Net Change",
    }
    return unavailable_fallbacks.get(requested, options[0])


def _render_heatmap_iframe(html: str, height: int) -> None:
    st.iframe(html, height=height)


def render_heatmap(
    S: np.ndarray,
    last_diff: np.ndarray | None,
    step_count: int,
    config: dict,
    vmax: float | None = None,
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
        vmax: Optional shared color scale; defaults to max(|S|) when omitted

    Side effects:
        - Renders HTML via st.iframe()
        - Clears st.session_state.last_diff after rendering (prevents re-animation)
    """
    r = S.shape[0]
    if vmax is None:
        vmax = max(abs(S).max(), 1e-6)
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

    .scale-note {{
        text-align: center;
        color: #666;
        margin-top: 8px;
        font-size: 12px;
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
    html += f'<div class="caption">{step_count} {sentence_plural} written</div>\n'

    # Add color legend
    html += _render_color_legend(vmax)

    # Render with dynamic height (adjust based on r)
    # Base: table (60px/row) + margins (60px) + caption (40px) + legend (100px) = ~260px overhead
    height = (r * 60) + 260
    _render_heatmap_iframe(html, height)

    # Clear diff to prevent re-animation on next rerun
    st.session_state.last_diff = None


def render_change_heatmap(
    matrix: np.ndarray,
    step_index: int,
    vmax: float,
    caption: str | None = None,
) -> None:
    """
    Render a change matrix (diff or component) as a static HTML heatmap.

    Similar to render_heatmap() but:
    - No pulse animation (static view of historical data)
    - Uses caller-provided vmax for cross-view color comparability
    - Displays custom caption (e.g., "Change from sentence #3")
    - Does NOT clear last_diff

    Args:
        matrix: Matrix to visualize (diff or component, r×r numpy array)
        step_index: Step index for caption (0-indexed internally)
        vmax: Shared color scale maximum (same legend across all views)
        caption: Optional custom caption (defaults to "Change from sentence #N")
    """
    r = matrix.shape[0]
    
    # Default caption
    if caption is None:
        step_num = step_index + 1  # Display as 1-indexed
        caption = f"Change to memory matrix after adding sentence #{step_num}"
    
    # CSS styles (no pulse animation - same as live view for consistency)
    html = """
<style>
    table.change-heatmap {
        border-collapse: collapse;
        margin: 20px auto;
        font-family: monospace;
    }
    
    td.change-cell {
        width: 60px;
        height: 60px;
        text-align: center;
        vertical-align: middle;
        border: 1px solid #ddd;
        font-size: 11px;
        font-weight: 500;
    }
    
    .change-caption {
        text-align: center;
        color: #666;
        margin-top: 15px;
        font-size: 13px;
        font-weight: 500;
    }
    
    .legend-container {
        margin-top: 30px;
        text-align: center;
    }
    
    .legend-bar {
        background: linear-gradient(to right, 
            #2166ac 0%, #67a9cf 25%, #f7f7f7 50%, #ef8a62 75%, #b2182b 100%);
        height: 20px;
        width: 320px;
        margin: 10px auto;
        border: 1px solid #ddd;
        border-radius: 3px;
    }
    
    .legend-labels {
        display: flex;
        justify-content: space-between;
        width: 320px;
        margin: 5px auto;
        font-size: 12px;
        color: #666;
        font-family: monospace;
    }

    .scale-note {
        text-align: center;
        color: #666;
        margin-top: 8px;
        font-size: 12px;
    }
</style>
"""

    # Build table
    html += '<table class="change-heatmap">\n'
    for i in range(r):
        html += "  <tr>\n"
        for j in range(r):
            value = matrix[i, j]
            color = _value_to_hex(value, vmax)
            
            html += f'    <td class="change-cell" style="background-color: {color};">'
            html += f"{value:.3f}</td>\n"
        html += "  </tr>\n"
    html += "</table>\n"

    # Add caption
    html += f'<div class="change-caption">{caption}</div>\n'

    # Add color legend
    html += _render_color_legend(vmax)

    # Render with dynamic height (same calculation as render_heatmap)
    height = (r * 60) + 260
    _render_heatmap_iframe(html, height)


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
    st.markdown(
        "A lightweight interactive testbed for the Online State of Associative "
        "Memory (OSAM) from delta-mem. Write sentences into the memory matrix, "
        "inspect per-sentence memory updates, and test retrieval behavior."
    )

    st.markdown(
        "Based on the paper [δ-mem: Efficient Online Memory for Large Language Models]({_PAPER_URL}) by Jingdi Lei, Di Zhang, Junxian Li, Weida Wang, Kaixuan Fan, Xiang Liu, Qihan Liu, Xiaoteng Ma, Baian Chen, Soujanya Poria"
    )

    link_col1, link_col2, _ = st.columns([1, 1, 8])
    with link_col1:
        st.markdown(
            f"[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=white)]"
            f"({_PAPER_URL})"
        )
    with link_col2:
        st.markdown(
            f"[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)]"
            f"({_GITHUB_URL})"
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
            # Requesting explicit confirmation for the reset
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
                        # Capture memory state before write for history
                        S_before = st.session_state.state.S.copy()
                        
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

                        # Save diff for heatmap pulse animation
                        st.session_state.last_diff = metadata["state_diff"]

                        # Store write history for per-sentence change visualization
                        st.session_state.write_history.append({
                            "step_index": metadata["step_index"],
                            "text": metadata["text"],
                            "diff": metadata["state_diff"],
                            "S_before": S_before,
                            "k": metadata["key"],
                            "v": metadata["value"],
                            "beta": metadata.get("beta_used", st.session_state.beta),
                        })
                        
                        # Reset selection to show live view for new write
                        st.session_state.selected_step = None

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
                    # Check if registry is empty (avoid wasted encoding work)
                    if len(st.session_state.registry.entries) == 0:
                        st.info(
                            "Memory is empty. Insert sentences first to query them."
                        )
                    else:
                        # Execute retrieval with loading spinner
                        with st.spinner("Searching memory..."):
                            # Encode query
                            query_embedding = st.session_state.ssw.encoder.encode(user_input)
                            
                            # Search registry
                            results = st.session_state.registry.search(
                                query_embedding=query_embedding,
                                state=st.session_state.state,
                                top_k=st.session_state.top_k
                            )
                        
                        # Store results and query in session state
                        st.session_state.last_results = results
                        st.session_state.last_query = user_input
                        
                        # Clear input field for next entry
                        st.session_state.current_input = ""
                        
                        # Rerun to show results
                        st.rerun()
        
        # Show success feedback after buttons (same location as error messages)
        if st.session_state.last_insert_step is not None:
            st.success("Sentence inserted")
            # Clear the flag so it doesn't show on every rerun
            st.session_state.last_insert_step = None
            st.session_state.last_insert_count = None

        st.write("---")

        # Retrieval Results Section (only visible when results exist)
        if st.session_state.last_results is not None:
            st.subheader("Retrieval Results")
            
            # Query header
            st.caption(f"Query: \"{st.session_state.last_query}\"")
            
            # Results list
            if len(st.session_state.last_results) == 0:
                st.info("No results found.")
            else:
                st.write("")  # Small spacing
                
                # Render each result
                for idx, result in enumerate(st.session_state.last_results, start=1):
                    _render_result_item(idx, result, is_top=(idx == 1))
            
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
            # "Back to live view" button (only when a step is selected)
            st.caption("Click a sentence in the registry to view a breakdown of its changes in the Memory Matrix.")
            if st.session_state.selected_step is not None:
                if st.button("← Back to Live View", key="back_to_live", use_container_width=True):
                    st.session_state.selected_step = None
                    st.rerun()
            
            # Display entries in reverse order (most recent first)
            for entry in reversed(registry.entries):
                step_num = entry.step_index + 1  # 0-indexed internally, display as 1-indexed
                text = entry.text
                
                # Truncate long sentences with ellipsis
                display_text = text if len(text) <= 60 else text[:57] + "..."
                
                # Check if this step is selected
                is_selected = (st.session_state.selected_step == entry.step_index)
                
                # Visual indicator for selected entry
                prefix = "▶ " if is_selected else ""
                button_label = f"{prefix}**#{step_num}** · {display_text}"
                
                # Create clickable button for each sentence
                button_type = "primary" if is_selected else "secondary"
                if st.button(
                    button_label,
                    key=f"select_step_{entry.step_index}",
                    use_container_width=True,
                    type=button_type,
                ):
                    st.session_state.selected_step = entry.step_index
                    st.rerun()

    # ---------------------------------------------------------------
    # Right column: Memory Matrix Visualization
    # ---------------------------------------------------------------
    with col_right:

        # Memory matrix heatmap visualization
        st.subheader("Memory Matrix")

        global_vmax = _compute_global_vmax(
            st.session_state.write_history,
            st.session_state.state.S,
        )
        
        # Branch: show change view if a step is selected, else live view
        if st.session_state.selected_step is not None:
            # Find the corresponding write history entry
            selected_entry = None
            for entry in st.session_state.write_history:
                if entry["step_index"] == st.session_state.selected_step:
                    selected_entry = entry
                    break
            
            if selected_entry is not None:
                # Show header indicating which sentence is being viewed
                step_num = selected_entry["step_index"] + 1
                st.caption(f"📊 Viewing change for sentence **#{step_num}**: \"{selected_entry['text'][:50]}...\"" if len(selected_entry['text']) > 50 else f"📊 Viewing change for sentence **#{step_num}**: \"{selected_entry['text']}\"")
                st.write("")  # Small spacing
                
                # Beta used at this write step (stored in write_history on insert)
                step_beta = selected_entry["beta"]
                st.caption(f"Write strength at this step: $\\boldsymbol{{\\beta_{{{step_num}}}}} = {step_beta:.2f}$")
                
                # Component selector (Phase 2: decomposition view)
                # For first sentence (step 0), S_{t-1} is zeros, so Retention and Erase are nil
                is_first_sentence = selected_entry["step_index"] == 0

                if is_first_sentence:
                    component_options = ["Net Change", "Write New Value", "Final State"]
                else:
                    component_options = [
                        "Retention",
                        "Erase Prediction",
                        "Write New Value",
                        "Final State",
                        "Net Change",
                    ]

                previous_view = st.session_state.selected_component_view
                resolved_view = _resolve_component_view(previous_view, component_options)
                if previous_view not in component_options and previous_view in (
                    "Retention",
                    "Erase Prediction",
                ):
                    st.caption(
                        "Retention and Erase are zero at step 0; showing "
                        f"**{resolved_view}** instead."
                    )

                if st.session_state.selected_component_view not in component_options:
                    st.session_state.selected_component_view = resolved_view
                if "global_component_view" not in st.session_state:
                    st.session_state.global_component_view = resolved_view
                elif st.session_state.global_component_view not in component_options:
                    st.session_state.global_component_view = resolved_view

                component_view = st.radio(
                    "Component view:",
                    component_options,
                    horizontal=True,
                    key="global_component_view",
                    help="View different components of the memory update equation",
                )
                st.session_state.selected_component_view = component_view
                
                # Compute selected component matrix and formula
                # step_index is 0-based; step_num = step_index + 1 is the post-write state S_t
                step_index = selected_entry["step_index"]
                S_before = selected_entry["S_before"]
                k = selected_entry["k"]
                v = selected_entry["v"]
                beta = step_beta
                
                if component_view == "Net Change":
                    matrix = selected_entry["diff"]
                    formula = f"S_{{{step_num}}} - S_{{{step_index}}}"
                    caption = f"Net change to memory from sentence #{step_num}"
                    
                elif component_view == "Retention":
                    matrix = (1 - beta) * S_before
                    formula = (
                        f"\\text{{Diag}}(\\lambda_{{{step_num}}}) S_{{{step_index}}} "
                        f"= (1-\\beta_{{{step_num}}}) S_{{{step_index}}}"
                    )
                    caption = "Retention component (scaled old state)"
                    
                elif component_view == "Erase Prediction":
                    prediction = S_before @ k
                    matrix = -beta * np.outer(prediction, k)
                    formula = (
                        f"-\\text{{Diag}}(\\beta_{{{step_num}}}) S_{{{step_index}}} "
                        f"\\mathbf{{k}}_{{{step_num}}} (\\mathbf{{k}}_{{{step_num}}})^\\top"
                    )
                    caption = "Erase prediction component"
                    
                elif component_view == "Write New Value":
                    matrix = beta * np.outer(v, k)
                    formula = (
                        f"+\\text{{Diag}}(\\beta_{{{step_num}}}) \\mathbf{{v}}_{{{step_num}}} "
                        f"(\\mathbf{{k}}_{{{step_num}}})^\\top"
                    )
                    caption = "Write new value component"

                elif component_view == "Final State":
                    retention = (1 - beta) * S_before
                    prediction = S_before @ k
                    erase = -beta * np.outer(prediction, k)
                    write = beta * np.outer(v, k)
                    matrix = retention + erase + write
                    formula = (
                        f"S_{{{step_num}}} = \\text{{Diag}}(\\lambda_{{{step_num}}}) S_{{{step_index}}} "
                        f"- \\text{{Diag}}(\\beta_{{{step_num}}}) S_{{{step_index}}} "
                        f"\\mathbf{{k}}_{{{step_num}}} (\\mathbf{{k}}_{{{step_num}}})^\\top "
                        f"+ \\text{{Diag}}(\\beta_{{{step_num}}}) \\mathbf{{v}}_{{{step_num}}} "
                        f"(\\mathbf{{k}}_{{{step_num}}})^\\top"
                    )
                    caption = f"Final memory state after sentence #{step_num} (sum of 3 components)"

                else:
                    st.error(f"Unknown component view: {component_view}")
                    st.stop()
                
                # Display the formula
                st.latex(formula)
                st.write("")  # Small spacing
                
                # Render the selected component heatmap
                render_change_heatmap(
                    matrix=matrix,
                    step_index=selected_entry["step_index"],
                    vmax=global_vmax,
                    caption=caption,
                )
            else:
                # Fallback if entry not found 
                st.warning(f"History entry for step {st.session_state.selected_step} not found.")
                render_heatmap(
                    S=st.session_state.state.S,
                    last_diff=st.session_state.last_diff,
                    step_count=st.session_state.ssw.step_counter,
                    config=CONFIG,
                    vmax=global_vmax,
                )
        else:
            # Live view: render current memory matrix with pulse animation
            st.caption("Viewing live memory matrix")
            render_heatmap(
                S=st.session_state.state.S,
                last_diff=st.session_state.last_diff,
                step_count=st.session_state.ssw.step_counter,
                config=CONFIG,
                vmax=global_vmax,
            )

        # ---------------------------------------------------------------
        # Standalone Reset Control
        # ---------------------------------------------------------------
        
        # Check if there's any data to reset
        _has_data = st.session_state.ssw.step_counter > 0 or len(st.session_state.registry.entries) > 0
        
        if not _has_data:
            # No data — show disabled-style info
            st.caption("Reset memory clears all stored sentences and the matrix.")
        elif not st.session_state.confirm_manual_reset:
            # Show reset button (centered with adjusted column ratios)
            _, c_btn, _ = st.columns([2, 1, 2])
            if c_btn.button("Reset Memory", key="manual_reset_btn", type="secondary", use_container_width=True):
                st.session_state.confirm_manual_reset = True
                st.rerun()
        else:
            # Confirmation prompt
            st.warning(
                f"**Confirm reset** — This will permanently clear the memory matrix "
                f"and all {st.session_state.ssw.step_counter} stored sentence(s). "
                f"This cannot be undone."
            )
            # Center the buttons with padding columns
            _, c_yes, c_no, _ = st.columns([1, 1, 1, 1])
            if c_yes.button("Yes, reset", key="manual_confirm_yes_btn", type="primary", use_container_width=True):
                # Reset with current r (don't change dimension)
                current_r = st.session_state.state.r
                _apply_reset(current_r)
                st.session_state.confirm_manual_reset = False
                st.rerun()
            if c_no.button("Cancel", key="manual_confirm_no_btn", use_container_width=True):
                st.session_state.confirm_manual_reset = False
                st.rerun()



if __name__ == "__main__":
    main()
