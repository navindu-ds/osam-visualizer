"""
Writing Strategies Module

Implements the Sequence-State Write (SSW) strategy
- One write event per submitted sentence
- Sentence encoded to embedding via chosen SentenceTransformer model
- Single delta-rule update applied to memory state

Paper reference: delta-mem Section 3.3 (delta-rule update) & 3.5 (Writing Granularity of Online State)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path
import yaml

from .encoder import (
    SentenceEncoderWrapper,
    ProjectionMatrix,
    project_to_key,
    project_to_value,
)
from .memory_state import OSAMState


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path (str, optional): Path to config file. If None, uses
                                     config/defaults.yaml relative to repo root.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if config_path is None:
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "config" / "defaults.yaml"

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class SequenceStateWrite:
    """
    Orchestrator for the Sequence-State Write (SSW) strategy.

    Coordinates the complete pipeline:
    1. Encode input text to embedding (via SentenceEncoderWrapper)
    2. Project to key/value vectors (via ProjectionMatrix + projection functions)
    3. Apply delta-rule update to memory state (via OSAMState.update)
    4. Return updated state and metadata for visualization

    All parameters are loaded from config/defaults.yaml and can be overridden
    by passing a custom config path or updating the Streamlit session state.

    Attributes:
        encoder (SentenceEncoderWrapper): Lazy-loaded sentence encoder.
        W_k (ProjectionMatrix): Key projection matrix.
        W_v (ProjectionMatrix): Value projection matrix.
        config (dict): Configuration dictionary.
        step_counter (int): Counter for write steps (incremented per execute).
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize SequenceStateWrite orchestrator.

        Loads configuration, initializes projection matrices, and prepares
        the sentence encoder (which is lazily loaded on first use).

        Args:
            config_path (str, optional): Path to config/defaults.yaml.
                                        If None, uses default location.

        Returns:
            None

        Raises:
            FileNotFoundError: If config file not found.
            KeyError: If required config keys missing.
            ValueError: If config values invalid.
        """
        self.config: dict = load_config(config_path)

        # Extract encoder config (required keys from defaults.yaml)
        encoder_config = self.config["encoder"]
        self.model_name: str = encoder_config["model_name"]
        embedding_dim: int = encoder_config["embedding_dim"]

        # Extract projections config (required keys from defaults.yaml)
        proj_config = self.config["projections"]
        input_dim: int = proj_config["input_dim"]
        output_dim: int = proj_config["output_dim"]
        init_scale: float = proj_config["init_scale"]

        # Validate config
        if input_dim < 1:
            raise ValueError(f"projections.input_dim must be positive; got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"projections.output_dim must be positive; got {output_dim}")
        if init_scale <= 0:
            raise ValueError(f"projections.init_scale must be positive; got {init_scale}")
        if embedding_dim != input_dim:
            raise ValueError(
                f"Encoder embedding_dim ({embedding_dim}) must match "
                f"projections.input_dim ({input_dim})"
            )

        # Initialize encoder (lazy-loads on first encode call)
        self.encoder: SentenceEncoderWrapper = SentenceEncoderWrapper(
            model_name=self.model_name, embedding_dim=embedding_dim
        )

        # Initialize projection matrices for key and value
        self.W_k: ProjectionMatrix = ProjectionMatrix(
            input_dim=input_dim, output_dim=output_dim, scale=init_scale
        )
        self.W_v: ProjectionMatrix = ProjectionMatrix(
            input_dim=input_dim, output_dim=output_dim, scale=init_scale
        )

        # Step counter for visualization
        self.step_counter: int = 0

    def execute(
        self, text: str, state: OSAMState, beta: Optional[float] = None
    ) -> Tuple[OSAMState, Dict]:
        """
        Execute a single write event: encode → project → update.

        Implements the Sequence-State Write (SSW) strategy 
        1. Encode text to embedding (dimension depends on encoder model)
        2. Project embedding to key vector via W_k + tanh + L2norm
        3. Project embedding to value vector via W_v (linear, no nonlinearity)
        4. Apply delta-rule update to memory: S_t = diag(λ) * S_{t-1} + diag(β) * (v - S_{t-1} * k) * k^T
        5. Increment step counter
        6. Return updated state and metadata for visualization

        Args:
            text (str): Input sentence to encode and write.
            state (OSAMState): Current memory state (not modified; returns new state).
            beta (float, optional): Write strength for this step. If None, uses default from config.
                                   Must be in (0, 1].

        Returns:
            Tuple[OSAMState, Dict]: Updated state and metadata dict containing:
                - 'embedding': Original sentence embedding (shape: embedding_dim)
                - 'key': Projected key vector (shape: output_dim)
                - 'value': Projected value vector (shape: output_dim)
                - 'state_diff': Memory state difference (shape: output_dim x output_dim)
                - 'step_index': Current step number (0-indexed)
                - 'text': Input text (for logging/display)
                - 'beta_used': Write strength value used for this step

        Raises:
            ValueError: If text is empty, state is invalid, or beta is out of bounds.
            TypeError: If text or state are wrong type.

        Paper reference: delta-mem Section 3.4 (delta-rule update)
        """
        # Input validation
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input text must be a non-empty string")
        if not isinstance(state, OSAMState):
            raise TypeError(
                f"state must be OSAMState instance; got {type(state).__name__}"
            )

        # Determine beta (write strength)
        if beta is None:
            beta = self.config["memory"]["write_strength"]
        if not (0 < beta <= 1):
            raise ValueError(f"beta must be in (0, 1]; got {beta}")

        # Step 1: Encode text to embedding
        embedding: np.ndarray = self.encoder.encode(text)

        # Step 2: Project to key and value
        key: np.ndarray = project_to_key(embedding, self.W_k)
        value: np.ndarray = project_to_value(embedding, self.W_v)

        # Step 3: Create a copy of input state (to avoid modifying it)
        # OSAMState.update() modifies self.S in-place, so we need a fresh instance
        state_copy = OSAMState(r=state.r, beta=state.beta)
        state_copy.S = state.S.copy()

        # Step 4: Apply delta-rule update to the copy
        # update() modifies state_copy.S in-place and returns (S_updated, diff)
        _, state_diff = state_copy.update(k=key, v=value, beta=beta)

        # Step 5: Increment step counter
        self.step_counter += 1

        # Step 6: Package metadata for visualization
        metadata: Dict = {
            "embedding": embedding,
            "key": key,
            "value": value,
            "state_diff": state_diff,
            "step_index": self.step_counter - 1,
            "text": text,
            "beta_used": beta,
        }

        return state_copy, metadata

    def reset_counter(self) -> None:
        """Reset the step counter to 0."""
        self.step_counter = 0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SequenceStateWrite(model={self.model_name}, "
            f"steps_executed={self.step_counter})"
        )
