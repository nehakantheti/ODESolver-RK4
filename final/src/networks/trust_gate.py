"""Adaptive trust gate for Parareal iteration control.

At each Parareal iteration, the trust gate decides — per time slab —
whether to accept the neural coarse propagator's prediction directly
or trigger an expensive fine RK4 correction.

Decision rule:
    action(n) = {
        accept y_hat_{n+1}^NN     if epsilon_n < tau   (NN is confident)
        run fine RK4 correction    if epsilon_n >= tau  (NN is uncertain)
    }

where:
    - epsilon_n is the confidence score from the coarse propagator
      (lower = more confident, like an error estimate)
    - tau is the gating threshold

The threshold tau can be:
    - Fixed: a scalar hyperparameter tuned on validation data.
    - Adaptive: decreases as Parareal iterations progress (the NN
      gets better after being corrected by fine solves).

Effect on Parareal:
    In early iterations, most slabs fail the gate -> full fine pass.
    In later iterations, the NN has been corrected and is more accurate ->
    more slabs pass the gate -> fewer fine solves -> faster convergence.

Safety guarantee:
    If the trust gate accepts a bad prediction, the next Parareal
    iteration's convergence check catches it (the error remains above
    tolerance).  The gate accelerates convergence but cannot cause
    *divergence* — Parareal's correction formula inherently damps errors.

Reference:
    - Inspired by the adaptive step-size DAL mechanism from
      Dherin et al. (Google, 2025).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class TrustGate:
    """Adaptive trust gate for selectively skipping Parareal fine passes.

    The gate examines the confidence score epsilon_n produced by the
    coarse propagator for each time slab and decides whether the NN's
    prediction is trustworthy enough to skip the expensive fine solver.

    Attributes:
        threshold: Current gating threshold tau.
        initial_threshold: The starting threshold value.
        decay_rate: Multiplicative decay applied to tau after each
                    Parareal iteration (tau *= decay_rate).
        min_threshold: Floor value — tau never goes below this.

    Example:
        >>> gate = TrustGate(initial_threshold=0.1)
        >>> confidence = torch.tensor([0.95, 0.80, 0.99, 0.60])
        >>> # Lower confidence score means higher error estimate
        >>> # Convert: error_estimate = 1 - confidence
        >>> error_estimates = 1 - confidence
        >>> mask = gate.should_run_fine(error_estimates)
        >>> # mask[i] = True means slab i needs fine correction
    """

    def __init__(
        self,
        initial_threshold: float = 0.05,
        decay_rate: float = 0.8,
        min_threshold: float = 0.001,
    ):
        """Initialise the trust gate.

        Args:
            initial_threshold: Starting threshold tau.  Slabs with
                              error estimate >= tau get fine correction.
                              Defaults to 0.05 (conservative — most slabs
                              get corrected initially).
            decay_rate: After each Parareal iteration, tau is multiplied
                       by this factor.  Values < 1 make the gate more
                       permissive over time.  Defaults to 0.8.
            min_threshold: Floor for tau to prevent it from reaching
                          zero.  Defaults to 0.001.
        """
        self.threshold = initial_threshold
        self.initial_threshold = initial_threshold
        self.decay_rate = decay_rate
        self.min_threshold = min_threshold

        logger.info(
            "TrustGate initialised: threshold=%.4f, decay=%.2f, min=%.4f",
            initial_threshold, decay_rate, min_threshold,
        )

    def should_run_fine(self, error_estimates: Tensor) -> Tensor:
        """Decide which slabs need fine RK4 correction.

        Compares each slab's error estimate against the current
        threshold.  Slabs with error >= threshold need correction;
        slabs below are trusted.

        Args:
            error_estimates: Per-slab error estimates (1 - confidence),
                            shape ``(n_slabs,)`` or ``(n_slabs, 1)``.
                            Values closer to 0 = more trustworthy.

        Returns:
            Boolean tensor of shape ``(n_slabs,)`` where ``True``
            means the slab needs fine correction.
        """
        error_flat = error_estimates.squeeze(-1)
        mask = error_flat >= self.threshold

        n_fine = mask.sum().item()
        n_skip = (~mask).sum().item()
        logger.debug(
            "TrustGate decision: %d slabs need fine correction, "
            "%d slabs trusted (threshold=%.4f)",
            n_fine, n_skip, self.threshold,
        )

        return mask

    def update_threshold(self, iteration: int) -> float:
        """Decay the threshold after a Parareal iteration.

        The threshold decreases geometrically, making the gate more
        permissive as the Parareal algorithm converges (the NN predictions
        improve after each fine correction round).

        Formula:
            tau = max(initial_threshold * decay_rate^iteration, min_threshold)

        Args:
            iteration: Current Parareal iteration number (0-indexed).

        Returns:
            The updated threshold value.
        """
        self.threshold = max(
            self.initial_threshold * (self.decay_rate ** iteration),
            self.min_threshold,
        )
        logger.debug(
            "TrustGate threshold updated: iteration=%d, tau=%.6f",
            iteration, self.threshold,
        )
        return self.threshold

    def reset(self) -> None:
        """Reset the threshold to its initial value.

        Call this before starting a new Parareal solve to ensure the gate
        begins in its conservative (most-slabs-corrected) state.
        """
        self.threshold = self.initial_threshold
        logger.debug("TrustGate reset to initial threshold=%.4f",
                      self.initial_threshold)

    def get_stats(self, error_estimates: Tensor) -> dict:
        """Compute gate statistics for monitoring and logging.

        Args:
            error_estimates: Per-slab error estimates, shape ``(n_slabs,)``.

        Returns:
            Dictionary with keys:
                - ``threshold``: current tau
                - ``n_slabs``: total number of slabs
                - ``n_trusted``: slabs passing the gate (no fine correction)
                - ``n_corrected``: slabs needing fine correction
                - ``trust_rate``: fraction of slabs trusted
                - ``mean_error``: mean error estimate across slabs
                - ``max_error``: maximum error estimate
        """
        error_flat = error_estimates.squeeze(-1)
        mask = self.should_run_fine(error_estimates)

        stats = {
            "threshold": self.threshold,
            "n_slabs": len(error_flat),
            "n_trusted": (~mask).sum().item(),
            "n_corrected": mask.sum().item(),
            "trust_rate": (~mask).float().mean().item(),
            "mean_error": error_flat.mean().item(),
            "max_error": error_flat.max().item(),
        }
        return stats
