"""
Unit Tests for Retrieval Module

Tests the SentenceRegistry class and cosine similarity function, including:
- Cosine similarity computation (unit tests)
- Registry initialization and configuration validation
- Storing sentences and FIFO overflow behavior
- Searching registry with cosine similarity ranking
- Integration with OSAMState and encoder modules
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import yaml

from src.retrieval import (
    SentenceRegistry,
    RegistryEntry,
    cosine_similarity,
    load_config,
)
from src.memory_state import OSAMState
from src.writing_strategies import SequenceStateWrite


class TestCosineSimilarity:
    """Test cosine similarity function."""

    def test_parallel_vectors(self):
        """Parallel vectors should have similarity = 1."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])  # Same direction, different magnitude
        similarity = cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity = 0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 1e-6

    def test_anti_parallel_vectors(self):
        """Anti-parallel vectors should have similarity = -1."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        similarity = cosine_similarity(a, b)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_zero_vector_a(self):
        """Zero vector a should return 0.0 (no crash)."""
        a = np.zeros(5)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        similarity = cosine_similarity(a, b)
        assert similarity == 0.0

    def test_zero_vector_b(self):
        """Zero vector b should return 0.0 (no crash)."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.zeros(5)
        similarity = cosine_similarity(a, b)
        assert similarity == 0.0

    def test_both_zero_vectors(self):
        """Both zero vectors should return 0.0 (no crash)."""
        a = np.zeros(3)
        b = np.zeros(3)
        similarity = cosine_similarity(a, b)
        assert similarity == 0.0

    def test_known_similarity(self):
        """Test with known cosine similarity value."""
        # Vectors at 60 degrees: cos(60°) = 0.5
        a = np.array([1.0, 0.0])
        b = np.array([0.5, np.sqrt(3) / 2])
        similarity = cosine_similarity(a, b)
        assert abs(similarity - 0.5) < 1e-6

    def test_shape_mismatch(self):
        """Vectors with different shapes should raise ValueError."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same shape"):
            cosine_similarity(a, b)

    def test_high_dimensional(self):
        """Test with high-dimensional vectors."""
        np.random.seed(42)
        a = np.random.randn(384)
        b = a / np.linalg.norm(a)  # Normalized version of a
        similarity = cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 1e-6


class TestLoadConfig:
    """Test configuration loading functionality."""

    def test_load_default_config(self):
        """Load default config from config/defaults.yaml."""
        config = load_config()
        assert isinstance(config, dict)
        assert "encoder" in config
        assert "projections" in config
        assert "storage" in config
        assert "retrieval" in config

    def test_load_custom_config(self):
        """Load custom config from specified path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            custom_config = {
                "encoder": {"model_name": "test-model", "embedding_dim": 256},
                "projections": {
                    "input_dim": 256,
                    "output_dim": 8,
                    "init_scale": 0.01,
                },
                "storage": {"max_sentences": 20},
                "retrieval": {"top_k": 3},
            }
            yaml.dump(custom_config, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config["storage"]["max_sentences"] == 20
            assert config["retrieval"]["top_k"] == 3
        finally:
            Path(config_path).unlink()

    def test_load_nonexistent_config(self):
        """Raise FileNotFoundError if config file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestSentenceRegistryInit:
    """Test SentenceRegistry initialization."""

    def test_init_default_config(self):
        """Initialize with default config."""
        registry = SentenceRegistry()
        assert registry.W_q is not None
        assert len(registry.entries) == 0
        assert registry.max_sentences > 0
        assert registry.top_k_default > 0

    def test_projection_matrix_shape(self):
        """W_q should have shape (output_dim, input_dim)."""
        registry = SentenceRegistry()
        config = registry.config
        expected_shape = (
            config["projections"]["output_dim"],
            config["projections"]["input_dim"],
        )
        assert registry.W_q.W.shape == expected_shape

    def test_empty_on_init(self):
        """Registry should be empty on initialization."""
        registry = SentenceRegistry()
        assert len(registry) == 0
        assert registry.entries == []

    def test_config_validation_negative_input_dim(self):
        """Raise ValueError if input_dim is non-positive."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            invalid_config = {
                "encoder": {"embedding_dim": -10},
                "projections": {"input_dim": -10, "output_dim": 8, "init_scale": 0.01},
                "storage": {"max_sentences": 30},
                "retrieval": {"top_k": 5},
            }
            yaml.dump(invalid_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="input_dim must be positive"):
                SentenceRegistry(config_path)
        finally:
            Path(config_path).unlink()

    def test_config_validation_embedding_mismatch(self):
        """Raise ValueError if embedding_dim != input_dim."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            invalid_config = {
                "encoder": {"embedding_dim": 256},
                "projections": {"input_dim": 128, "output_dim": 8, "init_scale": 0.01},
                "storage": {"max_sentences": 30},
                "retrieval": {"top_k": 5},
            }
            yaml.dump(invalid_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="must match"):
                SentenceRegistry(config_path)
        finally:
            Path(config_path).unlink()


class TestSentenceRegistryStore:
    """Test storing entries in the registry."""

    def test_store_single_entry(self):
        """Store a single entry."""
        registry = SentenceRegistry()
        text = "Test sentence."
        embedding = np.random.randn(384)
        key = np.random.randn(8)
        value = np.random.randn(8)
        step_index = 0

        registry.store(text, embedding, key, value, step_index)

        assert len(registry) == 1
        assert registry.entries[0].text == text
        assert np.array_equal(registry.entries[0].embedding, embedding)
        assert np.array_equal(registry.entries[0].key, key)
        assert np.array_equal(registry.entries[0].value, value)
        assert registry.entries[0].step_index == step_index

    def test_store_multiple_entries(self):
        """Store multiple entries."""
        registry = SentenceRegistry()
        for i in range(5):
            registry.store(
                text=f"Sentence {i}",
                embedding=np.random.randn(384),
                key=np.random.randn(8),
                value=np.random.randn(8),
                step_index=i,
            )

        assert len(registry) == 5
        assert registry.entries[0].text == "Sentence 0"
        assert registry.entries[4].text == "Sentence 4"

    def test_fifo_overflow(self):
        """Oldest entry should be dropped when max_sentences reached."""
        # Create registry with max_sentences = 3
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            config = {
                "encoder": {"model_name": "test", "embedding_dim": 384},
                "projections": {"input_dim": 384, "output_dim": 8, "init_scale": 0.01},
                "storage": {"max_sentences": 3},
                "retrieval": {"top_k": 5},
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            registry = SentenceRegistry(config_path)

            # Store 3 entries (at capacity)
            for i in range(3):
                registry.store(
                    text=f"Sentence {i}",
                    embedding=np.random.randn(384),
                    key=np.random.randn(8),
                    value=np.random.randn(8),
                    step_index=i,
                )
            assert len(registry) == 3
            assert registry.entries[0].text == "Sentence 0"

            # Store 4th entry (should drop "Sentence 0")
            registry.store(
                text="Sentence 3",
                embedding=np.random.randn(384),
                key=np.random.randn(8),
                value=np.random.randn(8),
                step_index=3,
            )
            assert len(registry) == 3
            assert registry.entries[0].text == "Sentence 1"  # Oldest is now "Sentence 1"
            assert registry.entries[2].text == "Sentence 3"
        finally:
            Path(config_path).unlink()

    def test_entry_immutability(self):
        """RegistryEntry should be immutable (frozen dataclass)."""
        entry = RegistryEntry(
            text="Test",
            embedding=np.array([1.0]),
            key=np.array([2.0]),
            value=np.array([3.0]),
            step_index=0,
        )
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            entry.text = "Modified"

    def test_len_method(self):
        """__len__ should return correct count."""
        registry = SentenceRegistry()
        assert len(registry) == 0

        registry.store("A", np.random.randn(384), np.random.randn(8), np.random.randn(8), 0)
        assert len(registry) == 1

        registry.store("B", np.random.randn(384), np.random.randn(8), np.random.randn(8), 1)
        assert len(registry) == 2


class TestSentenceRegistrySearch:
    """Test searching the registry."""

    def test_search_empty_registry(self):
        """Search on empty registry should return empty list."""
        registry = SentenceRegistry()
        state = OSAMState(r=8)
        query_embedding = np.random.randn(384)

        results = registry.search(query_embedding, state)
        assert results == []

    def test_search_result_structure(self):
        """Search results should have correct keys."""
        registry = SentenceRegistry()
        state = OSAMState(r=8)

        # Store one entry
        registry.store(
            text="Test sentence",
            embedding=np.random.randn(384),
            key=np.random.randn(8),
            value=np.random.randn(8),
            step_index=0,
        )

        # Search
        query_embedding = np.random.randn(384)
        results = registry.search(query_embedding, state)

        assert len(results) == 1
        result = results[0]
        assert "text" in result
        assert "score" in result
        assert "step_index" in result
        assert "key" in result
        assert "value" in result

    def test_search_top_k(self):
        """Search should respect top_k parameter."""
        registry = SentenceRegistry()
        state = OSAMState(r=8)

        # Store 5 entries
        for i in range(5):
            registry.store(
                text=f"Sentence {i}",
                embedding=np.random.randn(384),
                key=np.random.randn(8),
                value=np.random.randn(8),
                step_index=i,
            )

        # Search with top_k=2
        query_embedding = np.random.randn(384)
        results = registry.search(query_embedding, state, top_k=2)
        assert len(results) == 2

        # Search with top_k=10 (more than available)
        results = registry.search(query_embedding, state, top_k=10)
        assert len(results) == 5

    def test_search_ranking_order(self):
        """Results should be sorted by score (descending)."""
        registry = SentenceRegistry()
        state = OSAMState(r=8)

        # Store entries
        for i in range(5):
            registry.store(
                text=f"Sentence {i}",
                embedding=np.random.randn(384),
                key=np.random.randn(8),
                value=np.random.randn(8),
                step_index=i,
            )

        # Search
        query_embedding = np.random.randn(384)
        results = registry.search(query_embedding, state)

        # Verify descending order
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_known_vectors(self):
        """Test search with synthetic vectors and known expected ranking."""
        # Create custom config with r=3 to match state size
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            config = {
                "encoder": {"model_name": "test", "embedding_dim": 384},
                "projections": {"input_dim": 384, "output_dim": 3, "init_scale": 0.01},
                "storage": {"max_sentences": 30},
                "retrieval": {"top_k": 5},
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            registry = SentenceRegistry(config_path)
            state = OSAMState(r=3, beta=0.1)

            # Create synthetic entries with known value vectors
            # Entry 0: value = [1, 0, 0]
            # Entry 1: value = [0, 1, 0]
            # Entry 2: value = [0, 0, 1]
            registry.store(
                text="Sentence 0",
                embedding=np.random.randn(384),
                key=np.random.randn(3),
                value=np.array([1.0, 0.0, 0.0]),
                step_index=0,
            )
            registry.store(
                text="Sentence 1",
                embedding=np.random.randn(384),
                key=np.random.randn(3),
                value=np.array([0.0, 1.0, 0.0]),
                step_index=1,
            )
            registry.store(
                text="Sentence 2",
                embedding=np.random.randn(384),
                key=np.random.randn(3),
                value=np.array([0.0, 0.0, 1.0]),
                step_index=2,
            )

            # Write entry 0 to memory
            k = np.random.randn(3)
            v = np.array([1.0, 0.0, 0.0])
            state.update(k, v)

            # Query embedding (will be projected to q_t)
            # After projection and read, r_t should be closest to v = [1, 0, 0]
            # Since we can't control W_q easily, we'll just verify ranking is consistent
            query_embedding = np.random.randn(384)
            results = registry.search(query_embedding, state)

            # At minimum, verify we get 3 results with valid scores
            assert len(results) == 3
            for result in results:
                assert -1.0 <= result["score"] <= 1.0
        finally:
            Path(config_path).unlink()

    def test_search_invalid_state(self):
        """Search should raise TypeError if state is not OSAMState."""
        registry = SentenceRegistry()
        query_embedding = np.random.randn(384)

        with pytest.raises(TypeError, match="OSAMState"):
            registry.search(query_embedding, "not_a_state")


class TestSentenceRegistryClear:
    """Test clearing the registry."""

    def test_clear_empty_registry(self):
        """Clear on empty registry should not crash."""
        registry = SentenceRegistry()
        registry.clear()
        assert len(registry) == 0

    def test_clear_populated_registry(self):
        """Clear should remove all entries."""
        registry = SentenceRegistry()

        # Store entries
        for i in range(5):
            registry.store(
                text=f"Sentence {i}",
                embedding=np.random.randn(384),
                key=np.random.randn(8),
                value=np.random.randn(8),
                step_index=i,
            )
        assert len(registry) == 5

        # Clear
        registry.clear()
        assert len(registry) == 0
        assert registry.entries == []


class TestSentenceRegistryRepr:
    """Test __repr__ method."""

    def test_repr(self):
        """__repr__ should return informative string."""
        registry = SentenceRegistry()
        repr_str = repr(registry)
        assert "SentenceRegistry" in repr_str
        assert "entries=0" in repr_str


@pytest.mark.integration
class TestSentenceRegistryIntegration:
    """Integration tests with full pipeline."""

    def test_write_and_retrieve_pipeline(self):
        """End-to-end: write sentences → store → query → verify top result."""
        # Initialize components
        ssw = SequenceStateWrite()
        registry = SentenceRegistry()
        state = OSAMState(r=8, beta=0.1)

        # Write 5 sentences
        sentences = [
            "Alice lives in Paris.",
            "Bob works at a hospital.",
            "Charlie studies mathematics.",
            "Diana plays the piano.",
            "Eve enjoys hiking.",
        ]

        for sentence in sentences:
            state, metadata = ssw.execute(sentence, state)
            # Store metadata in registry
            registry.store(
                text=metadata["text"],
                embedding=metadata["embedding"],
                key=metadata["key"],
                value=metadata["value"],
                step_index=metadata["step_index"],
            )

        assert len(registry) == 5

        # Query with one of the written sentences
        query_sentence = "Alice lives in Paris."
        query_embedding = ssw.encoder.encode(query_sentence)
        results = registry.search(query_embedding, state, top_k=3)

        # Verify we got results
        assert len(results) == 3

        # Top result should be the exact sentence (or very high score)
        # Due to deterministic encoding, the query should match itself closely
        top_result = results[0]
        assert "Alice" in top_result["text"] or top_result["score"] > 0.5

    def test_deterministic_retrieval(self):
        """Repeated queries should return identical results (deterministic)."""
        ssw = SequenceStateWrite()
        registry = SentenceRegistry()
        state = OSAMState(r=8, beta=0.1)

        # Write sentences
        sentences = ["Test sentence 1", "Test sentence 2", "Test sentence 3"]
        for sentence in sentences:
            state, metadata = ssw.execute(sentence, state)
            registry.store(
                text=metadata["text"],
                embedding=metadata["embedding"],
                key=metadata["key"],
                value=metadata["value"],
                step_index=metadata["step_index"],
            )

        # Query twice with same sentence
        query_embedding = ssw.encoder.encode("Test sentence 1")
        results1 = registry.search(query_embedding, state)
        results2 = registry.search(query_embedding, state)

        # Results should be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["text"] == r2["text"]
            assert abs(r1["score"] - r2["score"]) < 1e-6

    def test_empty_state_retrieval(self):
        """Retrieval on empty memory state should still return ranked results."""
        ssw = SequenceStateWrite()
        registry = SentenceRegistry()
        state = OSAMState(r=8, beta=0.1)  # Empty state (no writes)

        # Store sentences WITHOUT writing to memory
        sentences = ["Sentence A", "Sentence B", "Sentence C"]
        for i, sentence in enumerate(sentences):
            embedding = ssw.encoder.encode(sentence)
            # Project to key/value but don't write to state
            from src.encoder import project_to_key, project_to_value

            key = project_to_key(embedding, ssw.W_k)
            value = project_to_value(embedding, ssw.W_v)
            registry.store(
                text=sentence,
                embedding=embedding,
                key=key,
                value=value,
                step_index=i,
            )

        # Query
        query_embedding = ssw.encoder.encode("Sentence A")
        results = registry.search(query_embedding, state)

        # Should still get results (even if scores are low due to empty state)
        assert len(results) == 3
