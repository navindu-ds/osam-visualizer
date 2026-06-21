"""
Unit Tests for Writing Strategies Module

Tests the Sequence-State Write (SSW) orchestrator, including:
- Configuration loading and validation
- Encoder and projection matrix initialization
- Complete encode → project → update pipeline
- Input validation and error handling
- Metadata structure and correctness
- Integration with OSAMState and encoder modules
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import yaml

from src.writing_strategies import SequenceStateWrite, load_config
from src.memory_state import OSAMState


class TestLoadConfig:
    """Test configuration loading functionality."""

    def test_load_default_config(self):
        """Load default config from config/defaults.yaml."""
        config = load_config()
        assert isinstance(config, dict)
        assert "encoder" in config
        assert "projections" in config
        assert "memory" in config

    def test_load_custom_config(self):
        """Load custom config from specified path."""
        # Create a temporary config file
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
                "memory": {"r": 8, "beta": 0.1},
            }
            yaml.dump(custom_config, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config["encoder"]["model_name"] == "test-model"
            assert config["encoder"]["embedding_dim"] == 256
        finally:
            Path(config_path).unlink()

    def test_load_nonexistent_config(self):
        """Raise FileNotFoundError if config file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_config_required_keys(self):
        """Verify config contains required keys."""
        config = load_config()
        assert "encoder" in config
        assert "projections" in config
        assert "memory" in config
        assert "encoder" in config
        assert "model_name" in config["encoder"]
        assert "input_dim" in config["projections"]
        assert "output_dim" in config["projections"]


class TestSequenceStateWriteInit:
    """Test SequenceStateWrite initialization."""

    def test_init_with_default_config(self):
        """Initialize with default config."""
        ssw = SequenceStateWrite()
        assert ssw.step_counter == 0
        assert ssw.encoder is not None
        assert ssw.W_k is not None
        assert ssw.W_v is not None
        assert isinstance(ssw.config, dict)

    def test_init_creates_projection_matrices(self):
        """Verify projection matrices are created with correct dimensions."""
        ssw = SequenceStateWrite()
        input_dim = ssw.config["projections"]["input_dim"]
        output_dim = ssw.config["projections"]["output_dim"]

        assert ssw.W_k.input_dim == input_dim
        assert ssw.W_k.output_dim == output_dim
        assert ssw.W_v.input_dim == input_dim
        assert ssw.W_v.output_dim == output_dim

    def test_init_with_custom_config(self):
        """Initialize with custom config path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            custom_config = {
                "encoder": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dim": 384,
                },
                "projections": {
                    "input_dim": 384,
                    "output_dim": 8,
                    "init_scale": 0.01,
                },
                "memory": {"r": 8, "beta": 0.1},
            }
            yaml.dump(custom_config, f)
            config_path = f.name

        try:
            ssw = SequenceStateWrite(config_path)
            assert ssw.config["projections"]["output_dim"] == 8
        finally:
            Path(config_path).unlink()

    def test_init_invalid_input_dim(self):
        """Raise ValueError if input_dim <= 0."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            bad_config = {
                "encoder": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dim": 384,
                },
                "projections": {
                    "input_dim": 0,  # Invalid
                    "output_dim": 8,
                    "init_scale": 0.01,
                },
                "memory": {"r": 8, "beta": 0.1},
            }
            yaml.dump(bad_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="input_dim must be positive"):
                SequenceStateWrite(config_path)
        finally:
            Path(config_path).unlink()

    def test_init_invalid_output_dim(self):
        """Raise ValueError if output_dim <= 0."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            bad_config = {
                "encoder": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dim": 384,
                },
                "projections": {
                    "input_dim": 384,
                    "output_dim": -1,  # Invalid
                    "init_scale": 0.01,
                },
                "memory": {"r": 8, "beta": 0.1},
            }
            yaml.dump(bad_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="output_dim must be positive"):
                SequenceStateWrite(config_path)
        finally:
            Path(config_path).unlink()

    def test_init_invalid_init_scale(self):
        """Raise ValueError if init_scale <= 0."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            bad_config = {
                "encoder": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dim": 384,
                },
                "projections": {
                    "input_dim": 384,
                    "output_dim": 8,
                    "init_scale": 0,  # Invalid
                },
                "memory": {"r": 8, "beta": 0.1},
            }
            yaml.dump(bad_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="init_scale must be positive"):
                SequenceStateWrite(config_path)
        finally:
            Path(config_path).unlink()

    def test_init_dimension_mismatch(self):
        """Raise ValueError if embedding_dim != input_dim."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            bad_config = {
                "encoder": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dim": 384,
                },
                "projections": {
                    "input_dim": 256,  # Mismatch
                    "output_dim": 8,
                    "init_scale": 0.01,
                },
                "memory": {"r": 8, "beta": 0.1},
            }
            yaml.dump(bad_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="embedding_dim.*must match"):
                SequenceStateWrite(config_path)
        finally:
            Path(config_path).unlink()

    def test_init_encoder_not_loaded_yet(self):
        """Encoder is not loaded until first encode call (lazy loading)."""
        ssw = SequenceStateWrite()
        assert ssw.encoder._encoder is None

    def test_step_counter_initialized_to_zero(self):
        """Step counter should start at 0."""
        ssw = SequenceStateWrite()
        assert ssw.step_counter == 0


class TestSequenceStateWriteExecute:
    """Test execute method (encode → project → update pipeline)."""

    def test_execute_with_valid_input(self):
        """Execute with valid text and state returns updated state and metadata."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        text = "Paris is the capital of France."

        updated_state, metadata = ssw.execute(text, state)

        assert isinstance(updated_state, OSAMState)
        assert isinstance(metadata, dict)

    def test_execute_metadata_structure(self):
        """Metadata dict contains all required keys."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        text = "Hello world."

        _, metadata = ssw.execute(text, state)

        required_keys = {
            "embedding",
            "key",
            "value",
            "state_diff",
            "step_index",
            "text",
            "beta_used",
        }
        assert required_keys.issubset(metadata.keys())

    def test_execute_embedding_shape(self):
        """Embedding in metadata has correct shape."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        embedding_dim = ssw.config["projections"]["input_dim"]

        _, metadata = ssw.execute("Test sentence.", state)

        assert metadata["embedding"].shape == (embedding_dim,)
        assert metadata["embedding"].dtype == np.float32

    def test_execute_key_shape(self):
        """Key in metadata has correct shape (output_dim)."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        output_dim = ssw.config["projections"]["output_dim"]

        _, metadata = ssw.execute("Test sentence.", state)

        assert metadata["key"].shape == (output_dim,)
        assert np.allclose(np.linalg.norm(metadata["key"]), 1.0, atol=1e-6)

    def test_execute_value_shape(self):
        """Value in metadata has correct shape (output_dim)."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        output_dim = ssw.config["projections"]["output_dim"]

        _, metadata = ssw.execute("Test sentence.", state)

        assert metadata["value"].shape == (output_dim,)

    def test_execute_state_diff_shape(self):
        """State diff in metadata has correct shape (output_dim × output_dim)."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        output_dim = ssw.config["projections"]["output_dim"]

        _, metadata = ssw.execute("Test sentence.", state)

        assert metadata["state_diff"].shape == (output_dim, output_dim)

    def test_execute_step_index(self):
        """Step index is correctly set."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        _, metadata1 = ssw.execute("First.", state)
        assert metadata1["step_index"] == 0

        # Use updated state for second write
        state2, _ = ssw.execute("Second.", state)
        _, metadata2 = ssw.execute("Third.", state2)
        assert metadata2["step_index"] == 2

    def test_execute_beta_in_metadata(self):
        """Beta used is recorded in metadata."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        _, metadata = ssw.execute("Test.", state, beta=0.2)
        assert metadata["beta_used"] == 0.2

    def test_execute_text_in_metadata(self):
        """Original text is stored in metadata."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        text = "Original text here."

        _, metadata = ssw.execute(text, state)
        assert metadata["text"] == text

    def test_execute_increments_step_counter(self):
        """Step counter increments after each execute."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        assert ssw.step_counter == 0
        ssw.execute("First.", state)
        assert ssw.step_counter == 1
        ssw.execute("Second.", state)
        assert ssw.step_counter == 2

    def test_execute_empty_text_raises_error(self):
        """Raise ValueError if text is empty."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        with pytest.raises(ValueError, match="non-empty string"):
            ssw.execute("", state)

    def test_execute_whitespace_only_text_raises_error(self):
        """Raise ValueError if text is whitespace only."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        with pytest.raises(ValueError, match="non-empty string"):
            ssw.execute("   ", state)

    def test_execute_none_text_raises_error(self):
        """Raise ValueError if text is None."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        with pytest.raises(ValueError):
            ssw.execute(None, state)

    def test_execute_non_string_text_raises_error(self):
        """Raise ValueError if text is not a string."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        with pytest.raises(ValueError):
            ssw.execute(123, state)

    def test_execute_none_state_raises_error(self):
        """Raise TypeError if state is None."""
        ssw = SequenceStateWrite()

        with pytest.raises(TypeError):
            ssw.execute("Test.", None)

    def test_execute_invalid_state_type_raises_error(self):
        """Raise TypeError if state is not OSAMState instance."""
        ssw = SequenceStateWrite()

        with pytest.raises(TypeError):
            ssw.execute("Test.", {"r": 8})

    def test_execute_beta_out_of_bounds_low(self):
        """Raise ValueError if beta <= 0."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        with pytest.raises(ValueError, match="beta must be"):
            ssw.execute("Test.", state, beta=0)

    def test_execute_beta_out_of_bounds_high(self):
        """Raise ValueError if beta > 1."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        with pytest.raises(ValueError, match="beta must be"):
            ssw.execute("Test.", state, beta=1.5)

    def test_execute_uses_default_beta_from_config(self):
        """If beta not provided, uses default from config."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        default_beta = ssw.config["memory"]["write_strength"]

        _, metadata = ssw.execute("Test.", state)
        assert metadata["beta_used"] == default_beta

    def test_execute_returns_new_state_instance(self):
        """Updated state is a new OSAMState instance."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        state_id_before = id(state)

        updated_state, _ = ssw.execute("Test.", state)
        state_id_after = id(updated_state)

        assert state_id_before != state_id_after

    def test_execute_does_not_modify_input_state(self):
        """Input state is not modified; update returns new state."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        state_copy = state.get_state().copy()

        ssw.execute("Test.", state)

        assert np.allclose(state.get_state(), state_copy)

    def test_execute_state_diff_matches_update(self):
        """State diff equals state_new - state_old."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        text = "Test sentence."

        # Get initial state
        state_before = state.get_state().copy()

        # Execute write
        updated_state, metadata = ssw.execute(text, state)

        # Verify diff = state_new - state_old
        state_after = updated_state.get_state()
        expected_diff = state_after - state_before

        assert np.allclose(metadata["state_diff"], expected_diff)


class TestSequenceStateWriteIntegration:
    """Integration tests for full writing strategy pipeline."""

    def test_sequential_writes(self):
        """Multiple sequential writes update state correctly."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]

        states = [state]
        for text in texts:
            state, _ = ssw.execute(text, states[-1])
            states.append(state)

        # Verify step counter
        assert ssw.step_counter == 3

        # Verify all states are different
        for i in range(len(states) - 1):
            for j in range(i + 1, len(states)):
                assert not np.allclose(
                    states[i].get_state(), states[j].get_state()
                )

    def test_deterministic_encoding(self):
        """Same text always produces same embedding."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        text = "This is a test."

        _, metadata1 = ssw.execute(text, state)
        _, metadata2 = ssw.execute(text, state)

        assert np.allclose(metadata1["embedding"], metadata2["embedding"])

    def test_different_texts_different_embeddings(self):
        """Different texts produce different embeddings."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        _, metadata1 = ssw.execute("First text.", state)
        _, metadata2 = ssw.execute("Second text.", state)

        assert not np.allclose(metadata1["embedding"], metadata2["embedding"])

    def test_key_is_unit_norm(self):
        """Key vector is always unit norm."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        texts = [
            "Short.",
            "This is a much longer sentence with more content.",
            "Mid length sentence.",
        ]

        for text in texts:
            _, metadata = ssw.execute(text, state)
            key_norm = np.linalg.norm(metadata["key"])
            assert np.isclose(key_norm, 1.0, atol=1e-6)

    def test_reset_counter(self):
        """reset_counter resets step counter to 0."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        ssw.execute("First.", state)
        ssw.execute("Second.", state)
        assert ssw.step_counter == 2

        ssw.reset_counter()
        assert ssw.step_counter == 0

    def test_different_beta_values(self):
        """Different beta values produce different state diffs."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)
        text = "Test sentence."

        _, metadata_beta01 = ssw.execute(text, state, beta=0.1)
        _, metadata_beta05 = ssw.execute(text, state, beta=0.5)

        # State diffs should be different
        assert not np.allclose(
            metadata_beta01["state_diff"], metadata_beta05["state_diff"]
        )

    def test_repr(self):
        """String representation is informative."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        ssw.execute("Test.", state)

        repr_str = repr(ssw)
        assert "SequenceStateWrite" in repr_str
        assert "steps_executed=1" in repr_str


class TestSequenceStateWriteEdgeCases:
    """Edge case tests for robustness."""

    def test_execute_with_special_characters(self):
        """Handles text with special characters."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        texts = [
            "Hello! @#$%^&*()",
            "Quote: 'test' and \"double quote\"",
            "Symbols: © ™ ® €",
        ]

        for text in texts:
            _, metadata = ssw.execute(text, state)
            assert metadata["text"] == text

    def test_execute_with_unicode(self):
        """Handles Unicode text (multilingual)."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        texts = [
            "Hello, 世界",  # Chinese
            "Bonjour, France!",  # French with accent
            "Привет мир",  # Russian
        ]

        for text in texts:
            _, metadata = ssw.execute(text, state)
            assert metadata["text"] == text

    def test_execute_with_very_long_text(self):
        """Handles very long input text."""
        ssw = SequenceStateWrite()
        state = OSAMState(r=8, beta=0.1)

        long_text = "This is a test sentence. " * 100
        _, metadata = ssw.execute(long_text, state)

        assert metadata["text"] == long_text
        assert metadata["embedding"].shape == (
            ssw.config["projections"]["input_dim"],
        )
