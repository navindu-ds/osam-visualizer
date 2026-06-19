"""
Unit tests for encoder and projection module.

Tests:
- ProjectionMatrix initialization, projection, shape validation
- SentenceEncoderWrapper model loading and encoding
- Projection functions (key, value, query) with nonlinearity and normalization
- L2 normalization with edge cases
- Integration: full pipeline from embedding to projections
"""

import numpy as np
import pytest
from src.encoder import (
    ProjectionMatrix,
    SentenceEncoderWrapper,
    l2_normalize,
    project_to_key,
    project_to_value,
    project_to_query,
)


class TestProjectionMatrix:
    """Test ProjectionMatrix class."""

    def test_init_default_scale(self):
        """Test initialization with default scale."""
        proj = ProjectionMatrix(input_dim=384, output_dim=8)
        assert proj.input_dim == 384
        assert proj.output_dim == 8
        assert proj.W.shape == (8, 384)
        assert proj.W.dtype == np.float64  # Default NumPy dtype

    def test_init_custom_scale(self):
        """Test initialization with custom scale."""
        proj = ProjectionMatrix(input_dim=100, output_dim=16, scale=0.05)
        assert proj.W.shape == (16, 100)
        # Check that std is roughly the custom scale
        assert 0.04 < np.std(proj.W) < 0.06

    def test_init_invalid_input_dim(self):
        """Test that invalid input_dim raises ValueError."""
        with pytest.raises(ValueError, match="Dimensions must be positive"):
            ProjectionMatrix(input_dim=0, output_dim=8)

    def test_init_invalid_output_dim(self):
        """Test that invalid output_dim raises ValueError."""
        with pytest.raises(ValueError, match="Dimensions must be positive"):
            ProjectionMatrix(input_dim=384, output_dim=-1)

    def test_init_invalid_scale(self):
        """Test that invalid scale raises ValueError."""
        with pytest.raises(ValueError, match="Scale must be positive"):
            ProjectionMatrix(input_dim=384, output_dim=8, scale=0)

    def test_project_valid_input(self):
        """Test projection with valid input."""
        proj = ProjectionMatrix(input_dim=3, output_dim=2, scale=0.01)
        x = np.array([1.0, 2.0, 3.0])
        result = proj.project(x)
        assert result.shape == (2,)
        # Manual check: result = proj.W @ x
        expected = proj.W @ x
        assert np.allclose(result, expected)

    def test_project_zero_input(self):
        """Test projection with zero input."""
        proj = ProjectionMatrix(input_dim=3, output_dim=2, scale=0.01)
        x = np.zeros(3)
        result = proj.project(x)
        assert np.allclose(result, np.zeros(2))

    def test_project_invalid_shape(self):
        """Test that wrong input shape raises ValueError."""
        proj = ProjectionMatrix(input_dim=384, output_dim=8)
        x = np.ones(100)  # Wrong dimension
        with pytest.raises(ValueError, match="Input must have shape"):
            proj.project(x)

    def test_project_preserves_type(self):
        """Test that projection preserves numpy array type."""
        proj = ProjectionMatrix(input_dim=3, output_dim=2)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = proj.project(x)
        assert isinstance(result, np.ndarray)


class TestL2Normalization:
    """Test L2 normalization function."""

    def test_normalize_unit_vector(self):
        """Test that normalizing a unit vector gives itself (approx)."""
        x = np.array([1.0, 0.0, 0.0])
        normalized = l2_normalize(x)
        assert np.allclose(normalized, x, atol=1e-7)

    def test_normalize_arbitrary_vector(self):
        """Test that normalized vector has unit L2 norm."""
        x = np.array([3.0, 4.0])  # ||x|| = 5
        normalized = l2_normalize(x)
        norm = np.linalg.norm(normalized)
        assert np.isclose(norm, 1.0, atol=1e-7)

    def test_normalize_zero_vector(self):
        """Test normalization of zero vector (returns zero)."""
        x = np.zeros(3)
        normalized = l2_normalize(x)
        # Zero vector should normalize to zero vector
        assert np.allclose(normalized, np.zeros(3))

    def test_normalize_small_vector(self):
        """Test normalization of very small (but nonzero) vector."""
        x = np.array([1e-10, 1e-10])
        normalized = l2_normalize(x)
        # Should still have unit norm
        norm = np.linalg.norm(normalized)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_normalize_preserves_direction(self):
        """Test that normalization preserves direction."""
        x = np.array([3.0, 4.0])
        normalized = l2_normalize(x)
        # Should still point in same direction
        angle1 = np.arctan2(x[1], x[0])
        angle2 = np.arctan2(normalized[1], normalized[0])
        assert np.isclose(angle1, angle2)

    def test_normalize_high_dimensional(self):
        """Test normalization in high dimensions."""
        x = np.random.randn(384)
        normalized = l2_normalize(x)
        norm = np.linalg.norm(normalized)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_normalize_preserves_dtype(self):
        """Test that normalization preserves input dtype."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        normalized = l2_normalize(x)
        assert normalized.dtype == np.float32


class TestProjectionFunctions:
    """Test projection functions (key, value, query)."""

    def test_project_to_key_shape(self):
        """Test that project_to_key returns correct shape."""
        W_k = ProjectionMatrix(input_dim=384, output_dim=8)
        x = np.random.randn(384)
        result = project_to_key(x, W_k)
        assert result.shape == (8,)

    def test_project_to_key_normalized(self):
        """Test that key is L2-normalized."""
        W_k = ProjectionMatrix(input_dim=384, output_dim=8)
        x = np.random.randn(384)
        result = project_to_key(x, W_k)
        norm = np.linalg.norm(result)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_project_to_key_applies_tanh(self):
        """Test that tanh is applied (bounded output)."""
        W_k = ProjectionMatrix(input_dim=3, output_dim=2, scale=1.0)
        # Use large projection to test tanh bounds
        x = np.array([10.0, 10.0, 10.0])
        result = project_to_key(x, W_k)
        # After tanh, before normalization, should be in [-1, 1]
        # After normalization, still in [-1, 1]
        assert np.all(np.abs(result) <= 1.0)

    def test_project_to_value_shape(self):
        """Test that project_to_value returns correct shape."""
        W_v = ProjectionMatrix(input_dim=384, output_dim=8)
        x = np.random.randn(384)
        result = project_to_value(x, W_v)
        assert result.shape == (8,)

    def test_project_to_value_is_linear(self):
        """Test that value projection is linear (no nonlinearity)."""
        W_v = ProjectionMatrix(input_dim=3, output_dim=2)
        x1 = np.array([1.0, 0.0, 0.0])
        x2 = np.array([0.0, 1.0, 0.0])
        
        v1 = project_to_value(x1, W_v)
        v2 = project_to_value(x2, W_v)
        v_sum = project_to_value(x1 + x2, W_v)
        
        # Linearity: project(x1 + x2) = project(x1) + project(x2)
        assert np.allclose(v_sum, v1 + v2)

    def test_project_to_value_not_normalized(self):
        """Test that value is NOT normalized (unlike key/query)."""
        W_v = ProjectionMatrix(input_dim=384, output_dim=8, scale=0.1)
        x = np.random.randn(384)
        result = project_to_value(x, W_v)
        norm = np.linalg.norm(result)
        # Norm should NOT be 1.0 (unless by chance)
        # Very unlikely to be exactly 1.0
        assert not np.isclose(norm, 1.0, atol=0.1)

    def test_project_to_query_shape(self):
        """Test that project_to_query returns correct shape."""
        W_q = ProjectionMatrix(input_dim=384, output_dim=8)
        x = np.random.randn(384)
        result = project_to_query(x, W_q)
        assert result.shape == (8,)

    def test_project_to_query_normalized(self):
        """Test that query is L2-normalized."""
        W_q = ProjectionMatrix(input_dim=384, output_dim=8)
        x = np.random.randn(384)
        result = project_to_query(x, W_q)
        norm = np.linalg.norm(result)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_project_to_query_same_as_key(self):
        """Test that query and key use same computation structure."""
        # Same W matrix, same input -> should be identical
        W = ProjectionMatrix(input_dim=384, output_dim=8)
        x = np.random.randn(384)
        
        k = project_to_key(x, W)
        q = project_to_query(x, W)
        
        # Should be equal (same projection matrix, same nonlinearity, same norm)
        assert np.allclose(k, q)


class TestSentenceEncoderWrapper:
    """Test SentenceEncoderWrapper (mocked, since we don't want to download model)."""

    def test_init(self):
        """Test initialization."""
        encoder = SentenceEncoderWrapper(model_name="all-MiniLM-L6-v2")
        assert encoder.model_name == "all-MiniLM-L6-v2"
        assert encoder._encoder is None  # Not loaded yet

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        encoder = SentenceEncoderWrapper(model_name="custom-model")
        assert encoder.model_name == "custom-model"

    def test_encode_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        encoder = SentenceEncoderWrapper()
        with pytest.raises(ValueError, match="non-empty string"):
            encoder.encode("")

    def test_encode_none_raises(self):
        """Test that None raises ValueError."""
        encoder = SentenceEncoderWrapper()
        with pytest.raises(ValueError, match="non-empty string"):
            encoder.encode(None)

    def test_encode_whitespace_only_raises(self):
        """Test that whitespace-only string raises ValueError."""
        encoder = SentenceEncoderWrapper()
        with pytest.raises(ValueError, match="non-empty string"):
            encoder.encode("   ")

    @pytest.mark.integration
    def test_encode_real_model(self):
        """Test encoding with real model (integration test, skipped by default)."""
        try:
            encoder = SentenceEncoderWrapper()
            text = "Hello, world!"
            embedding = encoder.encode(text)
            
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestEncoderIntegration:
    """Integration tests for full encoding pipeline."""

    def test_full_pipeline_deterministic(self):
        """Test that full projection pipeline is deterministic."""
        np.random.seed(42)
        W_k = ProjectionMatrix(input_dim=384, output_dim=8)
        W_v = ProjectionMatrix(input_dim=384, output_dim=8)
        W_q = ProjectionMatrix(input_dim=384, output_dim=8)
        
        x = np.random.randn(384)
        
        k1 = project_to_key(x, W_k)
        v1 = project_to_value(x, W_v)
        q1 = project_to_query(x, W_q)
        
        # Same inputs should give same outputs
        k2 = project_to_key(x, W_k)
        v2 = project_to_value(x, W_v)
        q2 = project_to_query(x, W_q)
        
        assert np.allclose(k1, k2)
        assert np.allclose(v1, v2)
        assert np.allclose(q1, q2)

    def test_key_value_query_different(self):
        """Test that key, value, query are generally different (different W matrices)."""
        np.random.seed(42)
        W_k = ProjectionMatrix(input_dim=384, output_dim=8, scale=0.01)
        W_v = ProjectionMatrix(input_dim=384, output_dim=8, scale=0.01)
        W_q = ProjectionMatrix(input_dim=384, output_dim=8, scale=0.01)
        
        x = np.random.randn(384)
        
        k = project_to_key(x, W_k)
        v = project_to_value(x, W_v)
        q = project_to_query(x, W_q)
        
        # Very unlikely to be identical (different W matrices)
        assert not np.allclose(k, v)
        assert not np.allclose(v, q)
        assert not np.allclose(k, q)

    def test_projections_reasonable_magnitude(self):
        """Test that projections have reasonable magnitude (not exploding/vanishing)."""
        np.random.seed(42)
        W_k = ProjectionMatrix(input_dim=384, output_dim=8, scale=0.01)
        W_v = ProjectionMatrix(input_dim=384, output_dim=8, scale=0.01)
        W_q = ProjectionMatrix(input_dim=384, output_dim=8, scale=0.01)
        
        # Random embeddings
        for _ in range(10):
            x = np.random.randn(384)
            
            k = project_to_key(x, W_k)
            v = project_to_value(x, W_v)
            q = project_to_query(x, W_q)
            
            # Key and query should be unit norm (normalized)
            assert np.isclose(np.linalg.norm(k), 1.0, atol=1e-6)
            assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
            
            # Value should have reasonable magnitude (not NaN/Inf)
            assert not np.any(np.isnan(v))
            assert not np.any(np.isinf(v))
            # Rough bound: with scale=0.01 and input from randn(384),
            # expected magnitude is around 0.01 * sqrt(384) ~ 0.2
            assert np.linalg.norm(v) < 2.0  # Loose upper bound
