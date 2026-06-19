"""
Encoder & Projection Module

Implements sentence encoding and projection of embeddings into the OSAM
key/value/query space. This module bridges language embeddings (from any
sentence-transformers model) to OSAM projections (configurable dimension).

The embedding model and dimensions are fully configurable via config/defaults.yaml:
- encoder.model_name: SentenceTransformer model to use
- projections.input_dim: Embedding dimension from the chosen model
- projections.output_dim: OSAM memory dimension (should match memory.size)

Supported embedding models:
  - all-MiniLM-L6-v2 (384-dim, default, fast, no API key)
  - all-mpnet-base-v2 (768-dim, higher quality)
  - BGE small/base/large (512/768/1024-dim, specialized for retrieval)
  - Other sentence-transformers models (any dimension supported)

Paper reference: delta-mem Section 3.1 (input processing & projections)
"""

import numpy as np
from typing import Optional
from pathlib import Path
import yaml


class ProjectionMatrix:
    """
    Learnable projection matrix W for linear transformations.

    Stores a weight matrix W of shape (output_dim, input_dim) initialized
    with small Gaussian noise (scale configurable, per LoRA practice).

    Attributes:
        W (np.ndarray): Weight matrix, shape (output_dim, input_dim).
        input_dim (int): Input vector dimension (from embedding model).
        output_dim (int): Output vector dimension (typically memory.size).
    """

    def __init__(
        self, input_dim: int, output_dim: int, scale: float = 0.01
    ) -> None:
        """
        Initialize projection matrix with small Gaussian weights.

        Args:
            input_dim (int): Dimension of input vectors (depends on embedding model).
                            E.g., 384 for all-MiniLM-L6-v2, 768 for all-mpnet-base-v2.
            output_dim (int): Dimension of output vectors (typically memory.size).
                             E.g., 8 for default OSAM memory dimension.
            scale (float): Standard deviation for Gaussian initialization.
                          Default 0.01 (small initialization per LoRA practice).

        Returns:
            None

        Note:
            Small initialization prevents numerical instability and allows
            gradual learning of appropriate projection scales. This applies
            to any input/output dimensions, regardless of embedding model.
        """
        if input_dim < 1 or output_dim < 1:
            raise ValueError(
                f"Dimensions must be positive; got input={input_dim}, "
                f"output={output_dim}"
            )
        if scale <= 0:
            raise ValueError(f"Scale must be positive; got {scale}")

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        # W: shape (output_dim, input_dim)
        self.W: np.ndarray = np.random.randn(output_dim, input_dim) * scale

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Apply projection: y = W @ x.

        Args:
            x (np.ndarray): Input vector, shape (input_dim,).

        Returns:
            np.ndarray: Projected vector, shape (output_dim,).

        Raises:
            ValueError: If x has incorrect shape.
        """
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"Input must have shape ({self.input_dim},); got {x.shape}"
            )
        return self.W @ x

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProjectionMatrix(input={self.input_dim}, "
            f"output={self.output_dim}, scale={np.std(self.W):.4f})"
        )


class SentenceEncoderWrapper:
    """
    Wrapper for sentence encoder model (any SentenceTransformer model).

    Lazily loads the SentenceTransformer model on first use and caches it.
    Returns sentence embeddings as numpy arrays. Dimension depends on the
    chosen model (e.g., 384 for all-MiniLM-L6-v2, 768 for all-mpnet-base-v2).

    In a Streamlit app, use @st.cache_resource decorator to avoid reloading
    the model on every page refresh.

    Attributes:
        model_name (str): Name of the SentenceTransformer model.
        embedding_dim (int): Output dimension of the embedding model.
        _encoder: Cached encoder instance (None until first use).
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384
    ) -> None:
        """
        Initialize encoder wrapper.

        Args:
            model_name (str): SentenceTransformer model identifier.
                             Default: all-MiniLM-L6-v2 (384-dim, fast, quality).
                             Other options:
                               - "sentence-transformers/all-mpnet-base-v2" (768-dim)
                               - "sentence-transformers/bge-small-en-v1.5" (384-dim)
                               - "sentence-transformers/bge-base-en-v1.5" (768-dim)
            embedding_dim (int): Expected output dimension of the model.
                                Default 384 (for all-MiniLM-L6-v2).
                                Must match the model's actual output dimension.

        Returns:
            None

        Note:
            Model is not loaded until encode() is first called. This allows
            Streamlit @st.cache_resource to intercept the loading.
            
            The embedding_dim is informational and used for validation.
            It should match the actual model output; mismatches will cause
            shape errors during projection.

        Example:
            # Use default model
            encoder = SentenceEncoderWrapper()
            
            # Use different model with custom dimension
            encoder = SentenceEncoderWrapper(
                model_name="sentence-transformers/all-mpnet-base-v2",
                embedding_dim=768
            )
        """
        self.model_name: str = model_name
        self.embedding_dim: int = embedding_dim
        self._encoder = None

    def _load_encoder(self):
        """Lazy load the encoder on first use."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single sentence into an embedding.

        Args:
            text (str): Input sentence to encode.

        Returns:
            np.ndarray: Sentence embedding, shape (embedding_dim,), dtype float32.
                       Dimension depends on the model (e.g., 384 for all-MiniLM-L6-v2).

        Raises:
            ValueError: If text is empty or None.
            RuntimeError: If encoder fails to load.

        Note:
            The SentenceTransformer model returns embeddings normalized to
            unit length by default. This module does NOT re-normalize here
            to preserve the model's output.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input text must be a non-empty string")

        self._load_encoder()

        # SentenceTransformer.encode returns numpy array
        embedding = self._encoder.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def __repr__(self) -> str:
        """String representation."""
        return f"SentenceEncoderWrapper(model={self.model_name})"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize a vector.

    Computes: x_norm = x / ||x||, with zero vector handling.

    Args:
        x (np.ndarray): Input vector of any shape.

    Returns:
        np.ndarray: Normalized vector, same shape as input, with unit L2 norm.
                   For zero input vector, returns zero vector.

    Note:
        Mathematically correct: handles zero vector by returning zero,
        not by adding epsilon to the norm. This preserves the actual
        L2 norm definition and prevents semantic distortion.
    """
    norm = np.linalg.norm(x)
    if norm == 0:
        return np.zeros_like(x, dtype=x.dtype)
    return x / norm


def project_to_key(x: np.ndarray, W_k: ProjectionMatrix) -> np.ndarray:
    """
    Project input to key space: k_t = L2norm(tanh(W_k @ x)).

    The key is a normalized, non-linear projection used to address memory
    locations (where to write/read). Nonlinearity increases expressiveness.

    Args:
        x (np.ndarray): Input embedding, shape (input_dim,).
        W_k (ProjectionMatrix): Key projection matrix.

    Returns:
        np.ndarray: Normalized key vector, shape (output_dim,).
                   Unit L2 norm.

    Paper reference: delta-mem Section 3.1 (key projection with nonlinearity)
    """
    projected = W_k.project(x)  # shape (output_dim,)
    nonlinear = np.tanh(projected)  # Bounded to [-1, 1]
    return l2_normalize(nonlinear)  # Unit norm


def project_to_value(x: np.ndarray, W_v: ProjectionMatrix) -> np.ndarray:
    """
    Project input to value space: v_t = W_v @ x.

    The value is a linear projection (no nonlinearity) of the input. It
    represents the semantic content to be written into memory.

    Args:
        x (np.ndarray): Input embedding, shape (input_dim,).
        W_v (ProjectionMatrix): Value projection matrix.

    Returns:
        np.ndarray: Value vector, shape (output_dim,).

    Paper reference: delta-mem Section 3.1 (value projection, linear)
    """
    return W_v.project(x)  # shape (output_dim,)


def project_to_query(x: np.ndarray, W_q: ProjectionMatrix) -> np.ndarray:
    """
    Project input to query space: q_t = L2norm(tanh(W_q @ x)).

    The query is a normalized, non-linear projection used to retrieve
    information from memory (what to read). Same nonlinearity as key for
    semantic consistency.

    Args:
        x (np.ndarray): Input embedding, shape (input_dim,).
        W_q (ProjectionMatrix): Query projection matrix.

    Returns:
        np.ndarray: Normalized query vector, shape (output_dim,).
                   Unit L2 norm.

    Paper reference: delta-mem Section 3.1 (query projection with nonlinearity)
    """
    projected = W_q.project(x)  # shape (output_dim,)
    nonlinear = np.tanh(projected)  # Bounded to [-1, 1]
    return l2_normalize(nonlinear)  # Unit norm


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
        # Assume repo root is parent of src/ (where this file lives)
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "config" / "defaults.yaml"

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
