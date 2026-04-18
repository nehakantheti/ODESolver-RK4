"""k-Factor residual prediction network for accelerating RK4 steps.

Instead of computing all four RK4 stages (k1, k2, k3, k4) sequentially
— each requiring one evaluation of the derivative function f — this
network predicts *corrections* to approximate k2, k3, k4 given an
exact k1.

How it works:
    1. Compute k1 = h * f(t_n, y_n) exactly (one f evaluation).
    2. Feed [k1, y_n, t_n, h] into the residual network.
    3. Network outputs corrections [delta_2, delta_3, delta_4].
    4. Approximate: k_hat_i = k1 + delta_i  for i = 2, 3, 4.

This replaces 3 sequential f evaluations with 1 NN forward pass,
which is especially valuable when f is computationally expensive
(e.g., involves a PDE spatial operator).

Why residual prediction:
    The corrections delta_i are typically small and smooth — they
    represent the *difference* between successive RK4 stages.  Learning
    residuals is fundamentally easier than learning absolute values,
    giving faster convergence and better generalisation.

Use-case gating:
    This module is optional.  For cheap f functions (e.g., simple
    polynomial ODEs), the NN overhead exceeds the savings.  The solver
    should gate activation based on f's computational cost.

Reference:
    - Othmane & Flaßkamp, "Neural network-enhanced integrators for
      simulating ODEs", arXiv:2504.05493, 2025.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """A single residual block: Linear → LayerNorm → SiLU + skip connection.

    Implements the pre-activation residual pattern where the skip
    connection adds the input directly to the transformed output.
    This helps gradient flow and stabilises training for deeper networks.

    Attributes:
        linear: Linear transformation layer.
        norm: Layer normalisation.
        activation: SiLU activation function.
    """

    def __init__(self, dim: int):
        """Initialise the residual block.

        Args:
            dim: Input and output dimension (must be equal for the
                 skip connection to work).
        """
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply transformation with skip connection.

        Args:
            x: Input tensor, shape ``(..., dim)``.

        Returns:
            Output tensor of the same shape: ``activation(norm(linear(x))) + x``.
        """
        return self.activation(self.norm(self.linear(x))) + x


class KFactorResidualNet(nn.Module):
    """Residual network predicting corrections delta_2, delta_3, delta_4.

    Given an exact k1 (computed from one evaluation of f), this network
    predicts the residual corrections needed to approximate the remaining
    three RK4 stages.

    Architecture:
        Input: [k1 (dim D), y_n (dim D), t_n (1), h (1)] → 2D + 2
            ↓
        Linear(2D+2, hidden) → LayerNorm → SiLU   (projection)
            ↓
        ResidualBlock(hidden)                       (residual layer)
            ↓
        ResidualBlock(hidden)                       (residual layer)
            ↓
        Linear(hidden, 3*D) → reshape to (3, D)
            ↓
        Output: [delta_2, delta_3, delta_4] each of dim D

    The final k-factor approximations are:
        k_hat_2 = k1 + delta_2
        k_hat_3 = k1 + delta_3
        k_hat_4 = k1 + delta_4

    Attributes:
        state_dim: Dimensionality of the ODE state vector.
        hidden_dim: Width of hidden layers.

    Example:
        >>> net = KFactorResidualNet(state_dim=2)
        >>> k1 = torch.randn(32, 2)
        >>> y_n = torch.randn(32, 2)
        >>> t_n = torch.randn(32, 1)
        >>> h = torch.randn(32, 1)
        >>> delta_2, delta_3, delta_4 = net(k1, y_n, t_n, h)
        >>> delta_2.shape  # (32, 2)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 96,
        n_residual_blocks: int = 2,
    ):
        """Initialise the k-factor residual network.

        Args:
            state_dim: Dimensionality of the ODE state (e.g., 2).
            hidden_dim: Width of hidden layers.  Defaults to 96
                        (multiple of 8 for GPU alignment, kept smaller
                        than the coarse propagator since this is a
                        lightweight accelerator).
            n_residual_blocks: Number of residual blocks.  Defaults to 2.
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Input: [k1, y_n, t_n, h] → 2 * state_dim + 2
        input_dim = 2 * state_dim + 2

        # Projection layer: map input to hidden dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(n_residual_blocks)]
        )

        # Output: 3 corrections of dim state_dim each
        self.output_layer = nn.Linear(hidden_dim, 3 * state_dim)

        logger.info(
            "KFactorResidualNet created: state_dim=%d, hidden_dim=%d, "
            "n_blocks=%d, total_params=%d",
            state_dim, hidden_dim, n_residual_blocks,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(
        self,
        k1: Tensor,
        y_n: Tensor,
        t_n: Tensor,
        h: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict residual corrections delta_2, delta_3, delta_4.

        How it works:
            1. Concatenates [k1, y_n, t_n, h] into a single feature vector.
            2. Projects to hidden dimension.
            3. Passes through residual blocks for feature extraction.
            4. Outputs 3 * state_dim values, reshaped to 3 correction vectors.

        Args:
            k1: First RK4 slope (exact), shape ``(batch, state_dim)``.
            y_n: Current state, shape ``(batch, state_dim)``.
            t_n: Current time, shape ``(batch, 1)``.
            h: Step size, shape ``(batch, 1)``.

        Returns:
            Tuple of three tensors (delta_2, delta_3, delta_4), each
            of shape ``(batch, state_dim)``.  The approximated k-factors
            are: k_hat_i = k1 + delta_i.
        """
        # Concatenate inputs
        x = torch.cat([k1, y_n, t_n, h], dim=-1)

        # Feature extraction
        features = self.projection(x)
        features = self.residual_blocks(features)

        # Predict corrections
        corrections = self.output_layer(features)

        # Reshape: (batch, 3 * state_dim) → 3 × (batch, state_dim)
        delta_2 = corrections[..., :self.state_dim]
        delta_3 = corrections[..., self.state_dim:2 * self.state_dim]
        delta_4 = corrections[..., 2 * self.state_dim:]

        return delta_2, delta_3, delta_4

    def predict_k_factors(
        self,
        k1: Tensor,
        y_n: Tensor,
        t_n: Tensor,
        h: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict full k-factors (k1 + delta) instead of raw deltas.

        Convenience method that adds k1 to each correction, giving the
        final approximated k2, k3, k4 values ready for the RK4 formula.

        Args:
            k1: First RK4 slope (exact), shape ``(batch, state_dim)``.
            y_n: Current state, shape ``(batch, state_dim)``.
            t_n: Current time, shape ``(batch, 1)``.
            h: Step size, shape ``(batch, 1)``.

        Returns:
            Tuple of (k_hat_2, k_hat_3, k_hat_4), each of shape
            ``(batch, state_dim)``.
        """
        delta_2, delta_3, delta_4 = self.forward(k1, y_n, t_n, h)
        return k1 + delta_2, k1 + delta_3, k1 + delta_4
