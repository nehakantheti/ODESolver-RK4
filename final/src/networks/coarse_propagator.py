"""Derivative-predicting neural coarse propagator for Parareal.

This module implements a neural network that learns the ODE vector field
f̂(y, t, θ) — the derivative dy/dt — rather than directly predicting
the next state y_{n+1}.  Integration is then performed externally:

    y_{n+1} = y_n + dt * f̂(y_n, t_n, θ)

Why derivative prediction is superior:
    1. **dt-independent**: Works for any step size without retraining.
    2. **Physics-aligned**: ODEs define dy/dt, so we learn what the
       ODE actually expresses.
    3. **Composable**: Can use Euler, RK2, or RK4 integration with the
       same learned f̂.
    4. **No extrapolation**: Multi-step propagation over long slabs
       naturally works because each step is small.

Architecture:
    Input:  [y_n (D), t_n (1), theta_ODE (P)]  →  D + 1 + P
    Trunk:  MLP with LayerNorm + SiLU + skip connections
    Output: f̂(y, t, θ) of shape (D,)

Reference:
    - Neural ODEs (Chen et al., 2018) — same philosophy of learning f̂
    - RandNet-Parareal, NeurIPS 2024
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class CoarsePropagatorNet(nn.Module):
    """Derivative-predicting meta-propagator for the Parareal coarse solver.

    Predicts the ODE vector field f̂(y, t, θ) conditioned on ODE parameters
    for cross-problem generalisation.  The caller performs integration
    (Euler, RK2, etc.) using the predicted derivative.

    Attributes:
        state_dim: Dimensionality of the ODE state vector.
        param_dim: Number of ODE parameters (theta_ODE).
        hidden_dim: Width of hidden layers.

    Example:
        >>> net = CoarsePropagatorNet(state_dim=2, param_dim=3)
        >>> y_n = torch.randn(16, 2)
        >>> t_n = torch.randn(16, 1)
        >>> theta = torch.randn(16, 3)
        >>> f_hat = net(y_n, t_n, theta)
        >>> f_hat.shape    # (16, 2) — learned derivative
        >>> # Integration: y_next = y_n + dt * f_hat
    """

    def __init__(
        self,
        state_dim: int,
        param_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
    ):
        """Initialise the derivative-predicting coarse propagator.

        Args:
            state_dim: Dimensionality of the ODE state (e.g., 2 for
                       position + velocity).
            param_dim: Number of ODE parameters in theta_ODE.
            hidden_dim: Width of each hidden layer.  Defaults to 128
                        (multiple of 8 for Tensor Core alignment).
            n_hidden_layers: Number of hidden layers.  Defaults to 3.
        """
        super().__init__()
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim

        # Input: [y_n (D), t_n (1), theta_ODE (P)]
        # NO delta_t — the network is dt-independent
        input_dim = state_dim + 1 + param_dim

        # Build shared trunk: input → hidden layers with skip connections
        layers = []
        prev_dim = input_dim
        for i in range(n_hidden_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Output: learned derivative f̂(y, t, θ)
        self.derivative_head = nn.Linear(hidden_dim, state_dim)

        # Initialise derivative head with small weights for stable start
        nn.init.xavier_uniform_(self.derivative_head.weight, gain=0.1)
        nn.init.zeros_(self.derivative_head.bias)

        logger.info(
            "CoarsePropagatorNet created (derivative mode): state_dim=%d, "
            "param_dim=%d, hidden_dim=%d, n_layers=%d, total_params=%d",
            state_dim, param_dim, hidden_dim, n_hidden_layers,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(
        self,
        y_n: Tensor,
        t_n: Tensor,
        theta_ode: Tensor,
    ) -> Tensor:
        """Predict the ODE derivative f̂(y, t, θ).

        Args:
            y_n: Current state, shape ``(batch, state_dim)``.
            t_n: Current time, shape ``(batch, 1)``.
            theta_ode: ODE parameter vector, shape ``(batch, param_dim)``.

        Returns:
            Predicted derivative f̂, shape ``(batch, state_dim)``.
        """
        # Concatenate: [y_n, t_n, theta_ode]
        x = torch.cat([y_n, t_n, theta_ode], dim=-1)

        # Feature extraction
        features = self.trunk(x)

        # Derivative prediction
        f_hat = self.derivative_head(features)

        return f_hat

    def predict(
        self,
        y_n: Tensor,
        t_n: float,
        theta_ode: Tensor,
    ) -> Tensor:
        """Convenience method for single-sample derivative prediction.

        Handles the scalar-to-tensor conversion for t_n and adds/removes
        the batch dimension automatically.

        Args:
            y_n: Current state, shape ``(state_dim,)`` (unbatched).
            t_n: Current time (scalar).
            theta_ode: ODE parameter vector, shape ``(param_dim,)``.

        Returns:
            Predicted derivative f̂, shape ``(state_dim,)``.
        """
        device = y_n.device

        # Add batch dimension
        y_batch = y_n.unsqueeze(0)
        t_batch = torch.tensor([[t_n]], dtype=torch.float32, device=device)
        theta_batch = theta_ode.unsqueeze(0)

        with torch.no_grad():
            f_hat = self.forward(y_batch, t_batch, theta_batch)

        # Remove batch dimension
        return f_hat.squeeze(0)

    def integrate_euler(
        self,
        y_n: Tensor,
        t_n: float,
        dt: float,
        theta_ode: Tensor,
    ) -> Tensor:
        """Forward Euler step using the learned derivative.

        y_{n+1} = y_n + dt * f̂(y_n, t_n, θ)

        Args:
            y_n: Current state, shape ``(state_dim,)``.
            t_n: Current time (scalar).
            dt: Time step.
            theta_ode: ODE parameter vector, shape ``(param_dim,)``.

        Returns:
            Next state, shape ``(state_dim,)``.
        """
        f_hat = self.predict(y_n, t_n, theta_ode)
        return y_n + dt * f_hat

    def integrate_rk2(
        self,
        y_n: Tensor,
        t_n: float,
        dt: float,
        theta_ode: Tensor,
    ) -> Tensor:
        """Heun's method (RK2) using the learned derivative.

        k1 = f̂(y_n, t_n)
        k2 = f̂(y_n + dt*k1, t_n + dt)
        y_{n+1} = y_n + dt/2 * (k1 + k2)

        Args:
            y_n: Current state, shape ``(state_dim,)``.
            t_n: Current time (scalar).
            dt: Time step.
            theta_ode: ODE parameter vector, shape ``(param_dim,)``.

        Returns:
            Next state, shape ``(state_dim,)``.
        """
        k1 = self.predict(y_n, t_n, theta_ode)
        y_mid = y_n + dt * k1
        k2 = self.predict(y_mid, t_n + dt, theta_ode)
        return y_n + (dt / 2.0) * (k1 + k2)
