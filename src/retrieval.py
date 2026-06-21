"""
Retrieval Module

Implements sentence registry and cosine similarity-based retrieval for OSAM.
The registry stores sentences with their projected vectors (key, value, embedding)
and enables querying the memory state to retrieve the most relevant stored sentences.

Paper reference: delta-mem Section 3.2 (reading from OSAM)
                In the paper, r_t feeds into LLM attention; here we use cosine
                similarity against stored value vectors for interpretability.
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from pathlib import Path
import yaml

from .encoder import ProjectionMatrix, project_to_query
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Cosine similarity = (a · b) / (||a|| * ||b||)
    Range: [-1, 1] where 1 = parallel, 0 = orthogonal, -1 = anti-parallel

    Explicit zero-vector handling (no epsilon masking):
    - If either vector is zero, returns 0.0 (no similarity, no crash)
    - This matches the l2_normalize convention from encoder.py (Step 1.3)

    Args:
        a (np.ndarray): First vector, shape (n,).
        b (np.ndarray): Second vector, shape (n,).

    Returns:
        float: Cosine similarity score in [-1, 1], or 0.0 if either is zero.

    Raises:
        ValueError: If vectors have different shapes.
    """
    if a.shape != b.shape:
        raise ValueError(f"Vectors must have same shape; got {a.shape} and {b.shape}")

    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Explicit zero-vector check (no epsilon)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Compute cosine similarity
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass(frozen=True)
class RegistryEntry:
    """
    Immutable record of a stored sentence and its projections.

    Stores all metadata returned by SequenceStateWrite.execute() for a single
    sentence. Frozen dataclass ensures immutability (no accidental mutations).

    Attributes:
        text (str): Original input sentence.
        embedding (np.ndarray): Sentence encoder output, shape (embedding_dim,).
        key (np.ndarray): Key projection k_t = L2norm(tanh(W_k @ embedding)).
        value (np.ndarray): Value projection v_t = W_v @ embedding.
        step_index (int): Write step number (0-indexed).
    """

    text: str
    embedding: np.ndarray
    key: np.ndarray
    value: np.ndarray
    step_index: int


class SentenceRegistry:
    """
    Registry for storing and retrieving sentences written to OSAM memory.

    Maintains a list of RegistryEntry objects (one per written sentence) and
    provides cosine similarity-based retrieval. When queried, the registry:
    1. Projects query embedding to q_t via W_q
    2. Reads memory: r_t = S @ q_t
    3. Scores each stored sentence: cosine_similarity(r_t, entry.value)
    4. Returns top-k ranked results

    Attributes:
        W_q (ProjectionMatrix): Query projection matrix (same dims as W_k, W_v).
        entries (list[RegistryEntry]): Stored sentences with their projections.
        max_sentences (int): Registry size limit (from config.storage.max_sentences).
        top_k_default (int): Default result count (from config.retrieval.top_k).
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize sentence registry.

        Loads configuration, initializes query projection matrix W_q, and
        prepares an empty entries list. W_q is initialized with the same
        dimensions and scale as W_k and W_v (from SequenceStateWrite).

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

        # Extract projections config (required keys from defaults.yaml)
        proj_config = self.config["projections"]
        input_dim: int = proj_config["input_dim"]
        output_dim: int = proj_config["output_dim"]
        init_scale: float = proj_config["init_scale"]

        # Extract encoder config for validation
        encoder_config = self.config["encoder"]
        embedding_dim: int = encoder_config["embedding_dim"]

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

        # Initialize query projection matrix
        self.W_q: ProjectionMatrix = ProjectionMatrix(
            input_dim=input_dim, output_dim=output_dim, scale=init_scale
        )

        # Extract storage and retrieval config
        storage_config = self.config["storage"]
        retrieval_config = self.config["retrieval"]
        self.max_sentences: int = storage_config["max_sentences"]
        self.top_k_default: int = retrieval_config["top_k"]

        # Initialize empty registry
        self.entries: List[RegistryEntry] = []

    def store(
        self,
        text: str,
        embedding: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        step_index: int,
    ) -> None:
        """
        Store a sentence and its projections in the registry.

        Creates an immutable RegistryEntry and appends to the entries list.
        If the registry is at max capacity (storage.max_sentences), removes
        the oldest entry (FIFO) before adding the new one.

        Args:
            text (str): Original input sentence.
            embedding (np.ndarray): Sentence encoder output, shape (embedding_dim,).
            key (np.ndarray): Key projection k_t, shape (output_dim,).
            value (np.ndarray): Value projection v_t, shape (output_dim,).
            step_index (int): Write step number (0-indexed).

        Returns:
            None
        """
        # FIFO overflow: drop oldest entry if at capacity
        if len(self.entries) >= self.max_sentences:
            self.entries.pop(0)

        # Create and store new entry
        entry = RegistryEntry(
            text=text,
            embedding=embedding,
            key=key,
            value=value,
            step_index=step_index,
        )
        self.entries.append(entry)

    def search(
        self,
        query_embedding: np.ndarray,
        state: OSAMState,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k sentences most similar to query via memory readout.

        Implements the OSAM read + cosine similarity retrieval pipeline:
        1. Project query embedding to query vector: q_t = L2norm(tanh(W_q @ x))
        2. Read from memory: r_t = S * q_t (constant-time, O(r²))
        3. Score each registry entry: cosine_similarity(r_t, entry.value)
        4. Sort by score (descending) and return top-k

        Args:
            query_embedding (np.ndarray): Query sentence embedding, shape (embedding_dim,).
            state (OSAMState): Current memory state to read from.
            top_k (int, optional): Number of results to return. If None, uses
                                  config.retrieval.top_k. If top_k > len(entries),
                                  returns all entries.

        Returns:
            List[Dict]: Ranked list of result dictionaries, each containing:
                - 'text' (str): Original sentence
                - 'score' (float): Cosine similarity score in [-1, 1]
                - 'step_index' (int): Write step number
                - 'key' (np.ndarray): Key projection
                - 'value' (np.ndarray): Value projection

        Raises:
            ValueError: If query_embedding has incorrect shape.
            TypeError: If state is not OSAMState instance.

        Paper reference: delta-mem Section 3.2 (reading from OSAM)
                        Read operation: r_t = S_{t-1} @ q_t
        """
        # Input validation
        if not isinstance(state, OSAMState):
            raise TypeError(
                f"state must be OSAMState instance; got {type(state).__name__}"
            )

        # Handle empty registry
        if len(self.entries) == 0:
            return []

        # Determine top_k
        if top_k is None:
            top_k = self.top_k_default
        top_k = min(top_k, len(self.entries))

        # Step 1: Project query embedding to query vector
        q_t: np.ndarray = project_to_query(query_embedding, self.W_q)

        # Step 2: Read from memory
        r_t: np.ndarray = state.read(q_t)

        # Step 3: Score each entry via cosine similarity
        results = []
        for entry in self.entries:
            score = cosine_similarity(r_t, entry.value)
            results.append(
                {
                    "text": entry.text,
                    "score": score,
                    "step_index": entry.step_index,
                    "key": entry.key,
                    "value": entry.value,
                }
            )

        # Step 4: Sort by score (descending) and return top-k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """
        Clear all entries from the registry.

        Resets the entries list to empty. Used when user resets memory state
        or starts a new session.

        Returns:
            None
        """
        self.entries = []

    def __len__(self) -> int:
        """Return the number of stored entries."""
        return len(self.entries)

    def __repr__(self) -> str:
        """String representation."""
        return f"SentenceRegistry(entries={len(self.entries)}, max={self.max_sentences})"
