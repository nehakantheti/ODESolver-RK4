"""k-Factor residual prediction network for accelerating RK4 steps.

Instead of computing all four RK4 stages (k1, k2, k3, k4) sequentially
— each requiring one evaluation of the derivative function f — this
network predicts *corrections* to approximate k2, k3, k4 given an
exact k1.

How it works:
    1. Compute k1 = h * f(t_n, y_n) exactly (one f evaluation).
    2. Feed [k1, y_n, t_n, h, theta_ode] into the residual network.
    3. Network outputs corrections [delta_2, delta_3, delta_4].
    4. Approximate: k_hat_i = k1 + delta_i  for i = 2, 3, 4.

This replaces 3 sequential f evaluations with 1 NN forward pass,
which is especially valuable when f is computationally expensive
(e.g., involves a PDE spatial operator).

Why theta_ode is included:
    The meta-learning approach conditions on ODE parameters so the
    same network generalises across the parameter family.  Without
    theta_ode, the k-factor net is blind to which system it's solving.

Why residual prediction:
    The corrections delta_i are typically small and smooth — they
    represent the *difference* between successive RK4 stages.  Learning
    residuals is fundamentally easier than learning absolute values,
    giving faster convergence and better generalisation.

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
    three RK4 stages, conditioned on ODE parameters theta_ode.

    Architecture:
        Input: [k1 (D), y_n (D), t_n (1), h (1), theta_ode (P)] → 2D + 2 + P
            ↓
        Linear(2D+2+P, hidden) → LayerNorm → SiLU   (projection)
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
        param_dim: Number of ODE parameters (theta_ODE).
        hidden_dim: Width of hidden layers.

    Example:
        >>> net = KFactorResidualNet(state_dim=2, param_dim=3)
        >>> k1 = torch.randn(32, 2)
        >>> y_n = torch.randn(32, 2)
        >>> t_n = torch.randn(32, 1)
        >>> h = torch.randn(32, 1)
        >>> theta = torch.randn(32, 3)
        >>> delta_2, delta_3, delta_4 = net(k1, y_n, t_n, h, theta)
        >>> delta_2.shape  # (32, 2)
    """

    def __init__(
        self,
        state_dim: int,
        param_dim: int = 0,
        hidden_dim: int = 96,
        n_residual_blocks: int = 2,
    ):
        """Initialise the k-factor residual network.

        Args:
            state_dim: Dimensionality of the ODE state (e.g., 2).
            param_dim: Number of ODE parameters in theta_ODE.
                      Defaults to 0 for backward compatibility.
            hidden_dim: Width of hidden layers.  Defaults to 96
                        (multiple of 8 for GPU alignment, kept smaller
                        than the coarse propagator since this is a
                        lightweight accelerator).
            n_residual_blocks: Number of residual blocks.  Defaults to 2.
        """
        super().__init__()
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim

        # Input: [k1, y_n, t_n, h, theta_ode] → 2 * state_dim + 2 + param_dim
        input_dim = 2 * state_dim + 2 + param_dim

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
            "KFactorResidualNet created: state_dim=%d, param_dim=%d, "
            "hidden_dim=%d, n_blocks=%d, total_params=%d",
            state_dim, param_dim, hidden_dim, n_residual_blocks,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(
        self,
        k1: Tensor,
        y_n: Tensor,
        t_n: Tensor,
        h: Tensor,
        theta_ode: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict residual corrections delta_2, delta_3, delta_4.

        Args:
            k1: First RK4 slope (exact), shape ``(batch, state_dim)``.
            y_n: Current state, shape ``(batch, state_dim)``.
            t_n: Current time, shape ``(batch, 1)``.
            h: Step size, shape ``(batch, 1)``.
            theta_ode: ODE parameter vector, shape ``(batch, param_dim)``.
                      Optional for backward compatibility (omit if param_dim=0).

        Returns:
            Tuple of three tensors (delta_2, delta_3, delta_4), each
            of shape ``(batch, state_dim)``.
        """
        # Concatenate inputs
        if theta_ode is not None and self.param_dim > 0:
            x = torch.cat([k1, y_n, t_n, h, theta_ode], dim=-1)
        else:
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
        theta_ode: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict full k-factors (k1 + delta) instead of raw deltas.

        Convenience method that adds k1 to each correction, giving the
        final approximated k2, k3, k4 values ready for the RK4 formula.

        Args:
            k1: First RK4 slope (exact), shape ``(batch, state_dim)``.
            y_n: Current state, shape ``(batch, state_dim)``.
            t_n: Current time, shape ``(batch, 1)``.
            h: Step size, shape ``(batch, 1)``.
            theta_ode: ODE parameter vector, shape ``(batch, param_dim)``.

        Returns:
            Tuple of (k_hat_2, k_hat_3, k_hat_4), each of shape
            ``(batch, state_dim)``.
        """
        delta_2, delta_3, delta_4 = self.forward(k1, y_n, t_n, h, theta_ode)
        return k1 + delta_2, k1 + delta_3, k1 + delta_4
