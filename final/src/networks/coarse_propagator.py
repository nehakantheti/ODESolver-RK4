"""Meta-propagator neural network for the Parareal coarse solver.

This module implements the coarse propagator network that predicts the
next state y_{n+1} given the current state y_n, time information, and
ODE parameters theta_ODE.

Architecture:
    A small MLP with parameter conditioning.  The network takes as input
    the concatenation [y_n, t_n, delta_t, theta_ODE] and outputs both
    the predicted next state y_hat_{n+1} and a scalar confidence score
    epsilon_n in [0, 1].

Key design feature — Meta-propagation:
    By conditioning on theta_ODE (the ODE parameter vector), a single
    trained network can generalise across a *family* of ODEs with
    different parameter values.  This avoids retraining per problem
    instance, unlike the PINN-Parareal approach (Ibrahim et al., 2023).

Training loss (semi-physics-informed):
    L = ||y_hat - y_fine||^2  +  lambda * ||y_hat' - f(t, y_hat)||^2
         <--  data loss  -->     <--   physics residual   -->

    The physics residual ensures the NN learns dynamically consistent
    trajectories, improving generalisation to unseen parameters.

Reference:
    - Ibrahim et al., "Parareal with a PINN as coarse propagator", 2023
    - RandNet-Parareal, NeurIPS 2024 (inspiration for fast coarse NN)
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class CoarsePropagatorNet(nn.Module):
    """Meta-propagator network for the Parareal coarse solver.

    Predicts the next ODE state over a coarse time step, conditioned on
    ODE parameters for cross-problem generalisation.

    The network has two output heads:
        1. **State head**: predicts y_hat_{n+1} (dim D)
        2. **Confidence head**: predicts epsilon_n in [0, 1] (scalar)

    Attributes:
        state_dim: Dimensionality of the ODE state vector.
        param_dim: Number of ODE parameters (theta_ODE).
        hidden_dim: Width of hidden layers (aligned to multiples of 8
                    for GPU Tensor Core efficiency).

    Example:
        >>> net = CoarsePropagatorNet(state_dim=2, param_dim=3)
        >>> y_n = torch.randn(16, 2)      # batch of 16, state dim 2
        >>> t_n = torch.randn(16, 1)
        >>> delta_t = torch.randn(16, 1)
        >>> theta = torch.randn(16, 3)    # 3 ODE params
        >>> y_hat, confidence = net(y_n, t_n, delta_t, theta)
        >>> y_hat.shape    # (16, 2)
        >>> confidence.shape  # (16, 1)
    """

    def __init__(
        self,
        state_dim: int,
        param_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
    ):
        """Initialise the meta-propagator network.

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

        # Input: [y_n (D), t_n (1), delta_t (1), theta_ODE (P)]
        input_dim = state_dim + 2 + param_dim

        # Build shared trunk: input → hidden layers
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

        # Output head 1: state prediction
        self.state_head = nn.Linear(hidden_dim, state_dim)

        # Output head 2: confidence score (scalar)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        logger.info(
            "CoarsePropagatorNet created: state_dim=%d, param_dim=%d, "
            "hidden_dim=%d, n_layers=%d, total_params=%d",
            state_dim, param_dim, hidden_dim, n_hidden_layers,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(
        self,
        y_n: Tensor,
        t_n: Tensor,
        delta_t: Tensor,
        theta_ode: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass: predict next state and confidence.

        Concatenates all inputs into a single feature vector, passes
        through the shared trunk, then splits into the state prediction
        and confidence score via separate output heads.

        Args:
            y_n: Current state, shape ``(batch, state_dim)``.
            t_n: Current time, shape ``(batch, 1)``.
            delta_t: Coarse time step, shape ``(batch, 1)``.
            theta_ode: ODE parameter vector, shape ``(batch, param_dim)``.

        Returns:
            Tuple of:
                - ``y_hat``: Predicted next state, shape ``(batch, state_dim)``.
                - ``confidence``: Confidence score in [0, 1], shape ``(batch, 1)``.
        """
        # Concatenate all inputs: [y_n, t_n, delta_t, theta_ode]
        x = torch.cat([y_n, t_n, delta_t, theta_ode], dim=-1)

        # Shared feature extraction
        features = self.trunk(x)

        # Dual-head output
        y_hat = self.state_head(features)
        confidence = self.confidence_head(features)

        return y_hat, confidence

    def predict(
        self,
        y_n: Tensor,
        t_n: float,
        delta_t: float,
        theta_ode: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Convenience method for single-sample inference.

        Handles the scalar-to-tensor conversion for t_n and delta_t,
        and adds/removes the batch dimension automatically.

        Args:
            y_n: Current state, shape ``(state_dim,)`` (unbatched).
            t_n: Current time (scalar).
            delta_t: Coarse time step (scalar).
            theta_ode: ODE parameter vector, shape ``(param_dim,)``.

        Returns:
            Tuple of:
                - ``y_hat``: Predicted next state, shape ``(state_dim,)``.
                - ``confidence``: Confidence score (scalar tensor).
        """
        device = y_n.device

        # Add batch dimension
        y_batch = y_n.unsqueeze(0)
        t_batch = torch.tensor([[t_n]], dtype=torch.float32, device=device)
        dt_batch = torch.tensor([[delta_t]], dtype=torch.float32, device=device)
        theta_batch = theta_ode.unsqueeze(0)

        with torch.no_grad():
            y_hat, confidence = self.forward(
                y_batch, t_batch, dt_batch, theta_batch
            )

        # Remove batch dimension
        return y_hat.squeeze(0), confidence.squeeze(0)
