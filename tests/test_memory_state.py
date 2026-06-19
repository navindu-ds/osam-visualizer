"""
Unit tests for OSAM Memory State Module (memory_state.py)

Tests verify:
- Correct initialization and shape
- Read operation (matrix-vector product)
- Delta-rule update with gating
- Diff computation (S_new - S_old)
- Reset functionality
- Input validation and error handling
- Numerical stability
"""

import numpy as np
import pytest
from src.memory_state import OSAMState


class TestOSAMStateInit:
    """Test memory state initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        state = OSAMState(r=8)
        assert state.r == 8
        assert state.beta == 0.1
        assert state.S.shape == (8, 8)
        assert np.allclose(state.S, np.zeros((8, 8)))

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        state = OSAMState(r=4, beta=0.2)
        assert state.r == 4
        assert state.beta == 0.2
        assert state.S.shape == (4, 4)

    def test_init_invalid_r(self):
        """Test that invalid r raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            OSAMState(r=0)
        with pytest.raises(ValueError, match="positive integer"):
            OSAMState(r=-5)

    def test_init_invalid_beta(self):
        """Test that invalid beta raises ValueError."""
        with pytest.raises(ValueError, match="in \\(0, 1\\]"):
            OSAMState(r=8, beta=0.0)
        with pytest.raises(ValueError, match="in \\(0, 1\\]"):
            OSAMState(r=8, beta=1.5)


class TestOSAMStateRead:
    """Test memory read operation."""

    def test_read_shape_and_type(self):
        """Test that read returns correct shape and type."""
        state = OSAMState(r=5)
        q = np.random.randn(5)
        result = state.read(q)
        assert result.shape == (5,)
        assert isinstance(result, np.ndarray)

    def test_read_zero_state(self):
        """Test read from zero state returns zero."""
        state = OSAMState(r=3)
        q = np.random.randn(3)
        result = state.read(q)
        assert np.allclose(result, np.zeros(3))

    def test_read_manual_calculation(self):
        """Test read against manual matrix-vector product."""
        state = OSAMState(r=3)
        # Set S to known values
        state.S = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        q = np.array([1.0, 0.0, 0.0])
        result = state.read(q)
        expected = np.array([1.0, 4.0, 7.0])
        assert np.allclose(result, expected)

    def test_read_invalid_shape(self):
        """Test that mismatched query shape raises error."""
        state = OSAMState(r=5)
        q = np.random.randn(4)  # Wrong shape
        with pytest.raises(ValueError, match="shape"):
            state.read(q)


class TestOSAMStateUpdate:
    """Test delta-rule update operation."""

    def test_update_returns_tuple(self):
        """Test that update returns (S_new, diff) tuple."""
        state = OSAMState(r=3, beta=0.1)
        k = np.array([1.0, 0.0, 0.0])
        v = np.array([0.5, 0.2, 0.1])
        result = state.update(k, v)
        assert isinstance(result, tuple)
        assert len(result) == 2
        S_new, diff = result
        assert S_new.shape == (3, 3)
        assert diff.shape == (3, 3)

    def test_update_zero_input(self):
        """Test update with zero key and value (no change expected)."""
        state = OSAMState(r=3, beta=0.1)
        k = np.zeros(3)
        v = np.zeros(3)
        S_new, diff = state.update(k, v)
        # With zero k and v, S should decay: S_new = (1-beta)*S_old + beta*(v - S*k)*k^T
        # Since S_old=0, v=0, k=0: S_new = 0
        assert np.allclose(S_new, np.zeros((3, 3)))
        assert np.allclose(diff, np.zeros((3, 3)))

    def test_update_simple_case(self):
        """Test update with known inputs and manual calculation."""
        state = OSAMState(r=2, beta=0.1)
        # Simple case: identity-like vectors
        k = np.array([1.0, 0.0])  # Normalized key (unit length)
        v = np.array([0.5, 0.2])  # Value vector
        beta = 0.1

        S_new, diff = state.update(k, v, beta=beta)

        # Manual calculation:
        # λ = 1 - 0.1 = 0.9
        # prediction = S_old @ k = 0
        # residual = v - 0 = [0.5, 0.2]
        # term1 = 0.9 * S_old = 0
        # outer_product = [0.5, 0.2]^T @ [1.0, 0.0] = [[0.5, 0], [0.2, 0]]
        # write_update = 0.1 * [[0.5, 0], [0.2, 0]] = [[0.05, 0], [0.02, 0]]
        # S_new = 0 + [[0.05, 0], [0.02, 0]] = [[0.05, 0], [0.02, 0]]

        expected_S = np.array([[0.05, 0.0], [0.02, 0.0]])
        assert np.allclose(S_new, expected_S, atol=1e-6)

    def test_update_diff_is_change(self):
        """Test that diff equals S_new - S_old."""
        state = OSAMState(r=4, beta=0.2)
        k = np.random.randn(4)
        v = np.random.randn(4)
        S_old = state.S.copy()
        S_new, diff = state.update(k, v)
        expected_diff = S_new - S_old
        assert np.allclose(diff, expected_diff)

    def test_update_retention_property(self):
        """
        Test retention property: well-learned facts cause small updates.

        If S already encodes a fact (S @ k ≈ v), the residual is small,
        so the update is small.
        """
        state = OSAMState(r=3, beta=0.1)
        # Pre-encode a fact: set S such that S @ k ≈ v
        k = np.array([1.0, 0.0, 0.0])
        v = np.array([0.5, 0.0, 0.0])
        state.S = np.outer(v, k)  # S @ k = v exactly

        S_old = state.S.copy()
        S_new, diff = state.update(k, v, beta=0.1)

        # Residual = v - S @ k = v - v = 0
        # So S_new = 0.9 * S_old + 0.1 * 0 = 0.9 * S_old
        expected_S = 0.9 * S_old
        expected_diff = 0.9 * S_old - S_old  # = -0.1 * S_old
        assert np.allclose(S_new, expected_S, atol=1e-6)
        assert np.allclose(diff, expected_diff, atol=1e-6)

    def test_update_invalid_k_shape(self):
        """Test that invalid k shape raises error."""
        state = OSAMState(r=5)
        k = np.random.randn(4)  # Wrong shape
        v = np.random.randn(5)
        with pytest.raises(ValueError, match="shape"):
            state.update(k, v)

    def test_update_invalid_v_shape(self):
        """Test that invalid v shape raises error."""
        state = OSAMState(r=5)
        k = np.random.randn(5)
        v = np.random.randn(4)  # Wrong shape
        with pytest.raises(ValueError, match="shape"):
            state.update(k, v)

    def test_update_invalid_beta(self):
        """Test that invalid beta raises error."""
        state = OSAMState(r=3)
        k = np.random.randn(3)
        v = np.random.randn(3)
        with pytest.raises(ValueError, match="in \\(0, 1\\]"):
            state.update(k, v, beta=0.0)
        with pytest.raises(ValueError, match="in \\(0, 1\\]"):
            state.update(k, v, beta=1.5)

    def test_update_uses_custom_beta(self):
        """Test that custom beta overrides default."""
        state = OSAMState(r=3, beta=0.1)
        k = np.array([1.0, 0.0, 0.0])
        v = np.array([0.5, 0.0, 0.0])

        # Update with custom beta
        S_new_custom, _ = state.update(k, v, beta=0.3)

        # Reset and update with default beta
        state.S = np.zeros((3, 3))
        S_new_default, _ = state.update(k, v)  # beta=0.1 (default)

        # With beta=0.3: S_new = 0.7*S_old + 0.3*(v - S_old @ k) @ k^T
        # With beta=0.1: S_new = 0.9*S_old + 0.1*(v - S_old @ k) @ k^T
        # Since S_old=0: 0.3*[0.5,0,0]@[1,0,0] vs 0.1*[0.5,0,0]@[1,0,0]
        # Expected ratio ≈ 3:1
        assert not np.allclose(S_new_custom, S_new_default)
        assert np.allclose(S_new_custom, 3 * S_new_default, atol=1e-6)


class TestOSAMStateReset:
    """Test reset functionality."""

    def test_reset_keeps_r(self):
        """Test that reset keeps current r."""
        state = OSAMState(r=5, beta=0.1)
        # Modify S
        state.S = np.random.randn(5, 5)
        state.reset()
        assert state.r == 5
        assert np.allclose(state.S, np.zeros((5, 5)))

    def test_reset_changes_r(self):
        """Test that reset with new r changes dimension."""
        state = OSAMState(r=4, beta=0.1)
        state.S = np.ones((4, 4))
        state.reset(r=3)
        assert state.r == 3
        assert state.S.shape == (3, 3)
        assert np.allclose(state.S, np.zeros((3, 3)))

    def test_reset_invalid_r(self):
        """Test that invalid new r raises error."""
        state = OSAMState(r=5)
        with pytest.raises(ValueError, match="positive integer"):
            state.reset(r=0)


class TestOSAMStateGetState:
    """Test state getter."""

    def test_get_state_returns_copy(self):
        """Test that get_state returns a copy, not a reference."""
        state = OSAMState(r=3)
        state.S = np.ones((3, 3))
        copy = state.get_state()
        copy[0, 0] = 999
        assert state.S[0, 0] == 1.0  # Original unchanged


class TestOSAMStateRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        state = OSAMState(r=8, beta=0.1)
        repr_str = repr(state)
        assert "OSAMState" in repr_str
        assert "r=8" in repr_str
        assert "beta=0.1" in repr_str


class TestOSAMStateIntegration:
    """Integration tests combining multiple operations."""

    def test_write_then_read(self):
        """Test writing a value then reading it back."""
        state = OSAMState(r=4, beta=0.1)
        k = np.array([1.0, 0.0, 0.0, 0.0])
        v = np.array([0.5, 0.2, 0.1, 0.0])

        # Write
        state.update(k, v)

        # Read with the same key
        result = state.read(k)

        # Result should be closer to v (but not exactly, due to imperfect encoding)
        # The network learns to approximate v along the key direction
        assert result[0] > 0  # Should have learned something

    def test_multiple_sequential_writes(self):
        """Test that S changes with each write."""
        state = OSAMState(r=3, beta=0.15)
        S_states = [state.S.copy()]

        # Multiple writes
        for i in range(3):
            k = np.random.randn(3)
            k = k / np.linalg.norm(k)  # Normalize
            v = np.random.randn(3)
            state.update(k, v)
            S_states.append(state.S.copy())

        # Check that S changed
        assert not np.allclose(S_states[0], S_states[1])
        assert not np.allclose(S_states[1], S_states[2])
        assert not np.allclose(S_states[2], S_states[3])

    def test_write_query_scenario(self):
        """
        Scenario: Write a fact, then query with similar input.

        This tests the core use case: encoding information and retrieving it.
        """
        state = OSAMState(r=5, beta=0.1)

        # Fact 1: Alice lives in Paris
        # Simplified as: k = [1, 0, 0, ...], v = [0.7, 0.2, 0.1, ...]
        k1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        v1 = np.array([0.7, 0.2, 0.1, 0.0, 0.0])
        state.update(k1, v1)

        # Query with same key
        result1 = state.read(k1)
        # Result should have learned the association: beta * v1[0] @ k1
        # S = 0.1 * v1 @ k1^T, so S @ k1 = 0.1 * v1 @ (k1^T @ k1) = 0.1 * v1 * 1 = 0.1 * v1
        # result1[0] = 0.1 * 0.7 = 0.07
        assert result1[0] > 0.05  # Should have learned ~0.07

        # Fact 2: Different key
        k2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.3, 0.5, 0.2, 0.0, 0.0])
        state.update(k2, v2)

        # Query with k1 again (should still have memory of it)
        result1_again = state.read(k1)
        # Due to retention (lambda = 0.9), old association decays but persists
        # Previous result was ~0.07, now multiplied by 0.9 ≈ 0.063
        assert result1_again[0] > 0.04


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
