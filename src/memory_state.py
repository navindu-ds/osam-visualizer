"""
OSAM Memory State Module

Core implementation of the Online State of Associative Memory (OSAM) component
from delta-mem (Lei et al., 2026, https://arxiv.org/abs/2605.12357).

The memory state S is an rxr matrix that stores compressed associations between
inputs and values using a delta-rule update mechanism. This module implements:
- Memory state initialization and read operations
- Gated delta-rule update (the core learning mechanism)
- State reset and numerical stability helpers

Paper reference: delta-mem Section 3.3 (OSAM state update)
"""

import numpy as np
from typing import Tuple


class OSAMState:
    """
    Online State of Associative Memory (OSAM).

    Maintains an r x r memory matrix S that is updated via a delta-rule mechanism.
    The matrix is read using matrix-vector multiplication (constant-time lookup)
    and written to using a gated delta-rule that balances retention of old
    information with integration of new information.

    Attributes:
        r (int): Dimension of the state matrix (r x r).
        S (np.ndarray): The r x r state matrix, initialized to zero.
        beta (float): Default write strength (0 < beta ≤ 1); can be overridden per-update.
    """

    def __init__(self, r: int, beta: float = 0.1) -> None:
        """
        Initialize the OSAM state.

        Args:
            r (int): Dimension of the state matrix. Default 8 per delta-mem paper.
            beta (float): Default write strength (control parameter for gating).
                         Default 0.1 as per paper. Range: (0, 1].
                         λ = 1 - β is the retention gate.

        Returns:
            None

        Paper reference: delta-mem Section 3.3 (OSAM initialization)
        """
        if not isinstance(r, int) or r < 1:
            raise ValueError(f"Memory size r must be a positive integer; got {r}")
        if not (0 < beta <= 1):
            raise ValueError(f"Write strength beta must be in (0, 1]; got {beta}")

        self.r: int = r
        self.beta: float = beta
        self.S: np.ndarray = np.zeros((r, r), dtype=np.float32)

    def read(self, q: np.ndarray) -> np.ndarray:
        """
        Read from memory using a query vector.

        Computes r_t = S @ q, a constant-time lookup regardless of history length.
        Memory access is O(r²), and not dependent on history length.

        Args:
            q (np.ndarray): Query vector of shape (r,). Typically a normalized
                           projection of the input: q_t = L2norm(tanh(W_q @ x)).

        Returns:
            np.ndarray: Read vector r_t of shape (r,). This is fed into retrieval
                       or fed back to the LLM (in full delta-mem system).

        Raises:
            ValueError: If q has incorrect shape.

        Paper reference: delta-mem Section 3.2 (reading from OSAM)
                        Equation: r_t = S_{t-1} @ q_t
        """
        if q.shape != (self.r,):
            raise ValueError(
                f"Query vector must have shape ({self.r},); got {q.shape}"
            )

        return self.S @ q

    def update(
        self, k: np.ndarray, v: np.ndarray, beta: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update memory using a key-value pair (delta-rule update).

        Implements the gated delta-rule update from delta-mem:

            S_t = diag(λ)*S_{t-1}  +  diag(β)*(v - S_{t-1}*k) * k^T

        Where:
            - λ = 1 - β (retention gate)
            - β (write strength) controls how much new info overwrites old
            - (v - S_{t-1} * k) is the prediction error (residual)
            - k^T is the outer product structure (writes along key direction)

        The update has three interpretable terms:
            1. diag(λ) * S_{t-1}:              Retain old state (decayed per-row)
            2. -diag(β) * S_{t-1} * k * k^T:   Erase old prediction along key direction
            3. +diag(β) * v * k^T:             Write new value along key direction

        Args:
            k (np.ndarray): Key vector of shape (r,). Normalized projection:
                           k_t = L2norm(tanh(W_k * x)).
            v (np.ndarray): Value vector of shape (r,). Projection without nonlinearity:
                           v_t = W_v * x.
            beta (float, optional): Write strength for this update. If None, uses self.beta.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - S_new (np.ndarray): Updated state matrix, shape (r, r).
                - diff (np.ndarray): Change in state, shape (r, r).
                                    diff = S_new - S_old. Used for visualization.

        Raises:
            ValueError: If k or v have incorrect shape, or beta is out of range.

        Paper reference: delta-mem Section 3.3 (delta-rule update with gating)
                        Implements the gated version with per-row gates (diag(β), diag(λ))
        """
        # Validate inputs
        if k.shape != (self.r,):
            raise ValueError(f"Key vector must have shape ({self.r},); got {k.shape}")
        if v.shape != (self.r,):
            raise ValueError(f"Value vector must have shape ({self.r},); got {v.shape}")

        # Use default beta if not provided
        if beta is None:
            beta = self.beta
        if not (0 < beta <= 1):
            raise ValueError(f"Write strength beta must be in (0, 1]; got {beta}")

        # Compute gates
        lamda = 1.0 - beta  # Retention gate
        # Note: In the paper, β and λ are per-row vectors. Here we use scalar β
        # for simplicity (all rows have the same write strength).

        # Save old state for diff computation
        S_old = self.S.copy()

        # Compute prediction error (residual): what the network doesn't yet encode
        # prediction = S * k (what the network currently predicts along key direction)
        prediction = self.S @ k  # shape (r,)
        residual = v - prediction  # shape (r,)

        # Compute the three update terms:
        # Term 1: Retention of old state (decayed per-row by λ)
        term1 = lamda * self.S  # shape (r, r)

        # Term 2 & 3: Write residual along key direction, scaled by β
        # This is the outer product: residual @ k^T
        # residual is shape (r,), k is shape (r,), so residual[:, None] @ k[None, :]
        # gives shape (r, r)
        outer_product = np.outer(residual, k)  # shape (r, r)
        write_update = beta * outer_product

        # Final update
        self.S = term1 + write_update

        # Compute diff for visualization
        diff = self.S - S_old

        return self.S, diff

    def reset(self, r: int | None = None) -> None:
        """
        Reset the memory state to zeros.

        Initializes S as a zero matrix. Used when the user resets the memory
        or changes the memory size r.

        Args:
            r (int, optional): New dimension for the state matrix. If None, keeps
                              current r. If provided, reinitializes S as (r, r).

        Returns:
            None

        Raises:
            ValueError: If new r is invalid.
        """
        if r is not None:
            if not isinstance(r, int) or r < 1:
                raise ValueError(f"Memory size r must be a positive integer; got {r}")
            self.r = r

        self.S = np.zeros((self.r, self.r), dtype=np.float32)

    def get_state(self) -> np.ndarray:
        """
        Return a copy of the current state matrix.

        Returns:
            np.ndarray: A copy of S, shape (r, r).
        """
        return self.S.copy()

    def __repr__(self) -> str:
        """String representation of the OSAM state."""
        return f"OSAMState(r={self.r}, beta={self.beta}, S.shape={self.S.shape})"
