"""Adaptive trust gate for Parareal iteration control.

At each Parareal iteration, the trust gate decides — per time slab —
whether to accept the current solution and skip the expensive fine RK4
correction, or run it.

Decision rule (convergence-based):
    action(n) = {
        SKIP fine correction    if slab n is "locked" (already converged)
        RUN  fine correction    otherwise
    }

A slab is locked when its per-slab correction magnitude
``|U_{n}^{k+1} - U_{n}^{k}|`` falls below a threshold for a
configurable number of consecutive iterations.  This is the standard
approach in Parareal literature — once a time slab has converged to
the fine-solver accuracy, re-solving it is wasted work.

Previous approach (v1, broken):
    Used the NN's confidence head ``epsilon_n``.  However, the
    confidence head was never supervised during training (loss only
    trained the state head), so ``epsilon_n ≈ 0.5`` always (random
    sigmoid), causing ``error_estimate = 1 - 0.5 = 0.5 >> threshold``
    for all slabs → trust_rate = 0%.

Current approach (v2, convergence-based):
    Uses the *actual* per-slab change ``|ΔU_n|`` from the previous
    Parareal iteration.  This directly measures how much each slab's
    solution moved — a slab that didn't move has converged and can be
    safely skipped.

Effect on Parareal:
    In early iterations, all slabs have large corrections → all run fine.
    In later iterations, leading slabs converge first → skipped → faster.
    The tail slabs (near t_end) converge last.

Safety guarantee:
    If a locked slab's upstream boundary changes (due to corrections
    propagating forward), the slab is automatically unlocked because
    its next correction will be non-zero.

Reference:
    - Lions, Maday, Turinici (2001): "converged slabs need not be resolved"
    - RandNet-Parareal, NeurIPS 2024: adaptive slab selection
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class TrustGate:
    """Convergence-based trust gate for selectively skipping fine passes.

    Tracks per-slab correction magnitudes across Parareal iterations.
    Once a slab's correction drops below ``lock_threshold`` for
    ``lock_patience`` consecutive iterations, the slab is "locked" and
    its fine solve is skipped.

    Attributes:
        lock_threshold: Correction magnitude below which a slab is
                       considered converged.
        lock_patience: Number of consecutive sub-threshold iterations
                      required before locking a slab.
        locked: Boolean tensor marking which slabs are locked.
        streak: Per-slab count of consecutive sub-threshold iterations.

    Example:
        >>> gate = TrustGate(lock_threshold=1e-4, lock_patience=1)
        >>> # After iteration k, per-slab changes are:
        >>> slab_changes = torch.tensor([1e-2, 3e-5, 5e-6, 2e-1])
        >>> gate.update_locks(slab_changes)
        >>> gate.locked   # tensor([False, True, True, False])
        >>> mask = gate.should_run_fine(slab_changes)
        >>> mask          # tensor([True, False, False, True])
    """

    def __init__(
        self,
        lock_threshold: float = 1e-4,
        lock_patience: int = 1,
        # Legacy parameters (kept for backward-compatibility)
        initial_threshold: float = 0.05,
        decay_rate: float = 0.8,
        min_threshold: float = 0.001,
    ):
        """Initialise the trust gate.

        Args:
            lock_threshold: Per-slab correction magnitude below which
                           the slab is considered converged.
            lock_patience: How many consecutive iterations below
                          threshold before locking.  1 = lock immediately.
            initial_threshold: (Legacy, unused) Kept for API compatibility.
            decay_rate: (Legacy, unused) Kept for API compatibility.
            min_threshold: (Legacy, unused) Kept for API compatibility.
        """
        self.lock_threshold = lock_threshold
        self.lock_patience = lock_patience
        # Per-slab tracking (initialised on first call)
        self.locked: Optional[Tensor] = None
        self.streak: Optional[Tensor] = None
        self._n_slabs: int = 0

        # Legacy attributes (for backward-compat with old call sites)
        self.threshold = initial_threshold
        self.initial_threshold = initial_threshold
        self.decay_rate = decay_rate
        self.min_threshold = min_threshold

        logger.info(
            "TrustGate initialised: lock_threshold=%.2e, patience=%d",
            lock_threshold, lock_patience,
        )

    def _ensure_init(self, n_slabs: int, device: torch.device) -> None:
        """Lazily create per-slab tracking tensors."""
        if self.locked is None or self._n_slabs != n_slabs:
            self.locked = torch.zeros(n_slabs, dtype=torch.bool, device=device)
            self.streak = torch.zeros(n_slabs, dtype=torch.long, device=device)
            self._n_slabs = n_slabs

    def update_locks(self, slab_changes: Tensor) -> None:
        """Update per-slab lock status based on correction magnitudes.

        Call this AFTER computing ``|U^{k+1} - U^k|`` per slab.

        Args:
            slab_changes: Per-slab max absolute change, shape ``(n_slabs,)``.
        """
        n_slabs = slab_changes.shape[0]
        self._ensure_init(n_slabs, slab_changes.device)

        # Which slabs had small corrections this iteration?
        below = slab_changes < self.lock_threshold

        # Update streak counters
        self.streak[below] += 1
        self.streak[~below] = 0  # reset if correction was large

        # Lock slabs that have been below threshold for enough iterations
        newly_locked = (~self.locked) & (self.streak >= self.lock_patience)
        if newly_locked.any():
            self.locked[newly_locked] = True
            locked_ids = newly_locked.nonzero(as_tuple=True)[0].tolist()
            logger.info(
                "TrustGate: locked slabs %s (change < %.2e for %d iters)",
                locked_ids, self.lock_threshold, self.lock_patience,
            )

        # Unlock slabs that had large corrections (upstream changed)
        reopen = self.locked & (~below)
        if reopen.any():
            self.locked[reopen] = False
            self.streak[reopen] = 0
            reopened_ids = reopen.nonzero(as_tuple=True)[0].tolist()
            logger.info(
                "TrustGate: unlocked slabs %s (correction grew)",
                reopened_ids,
            )

    def should_run_fine(self, error_estimates: Tensor) -> Tensor:
        """Decide which slabs need fine RK4 correction.

        This is now based on lock status, NOT on error_estimates.
        The ``error_estimates`` argument is kept for API compatibility
        but only used for initialisation sizing.

        Args:
            error_estimates: Per-slab values, shape ``(n_slabs,)``.
                            Used only for tensor size/device.

        Returns:
            Boolean tensor where ``True`` = needs fine correction.
        """
        n_slabs = error_estimates.squeeze(-1).shape[0]
        self._ensure_init(n_slabs, error_estimates.device)
        # Run fine on all non-locked slabs
        return ~self.locked

    def update_threshold(self, iteration: int) -> float:
        """Legacy method — no-op in convergence-based mode."""
        return self.lock_threshold

    def reset(self) -> None:
        """Reset all locks for a new Parareal solve."""
        self.locked = None
        self.streak = None
        self._n_slabs = 0
        logger.debug("TrustGate reset: all locks cleared")

    def get_stats(self, error_estimates: Tensor) -> dict:
        """Compute gate statistics for monitoring and logging.

        Args:
            error_estimates: Per-slab values, shape ``(n_slabs,)``.

        Returns:
            Dictionary with trust gate metrics.
        """
        n_slabs = error_estimates.squeeze(-1).shape[0]
        self._ensure_init(n_slabs, error_estimates.device)

        n_locked = self.locked.sum().item()
        stats = {
            "threshold": self.lock_threshold,
            "n_slabs": n_slabs,
            "n_trusted": int(n_locked),
            "n_corrected": int(n_slabs - n_locked),
            "trust_rate": n_locked / max(n_slabs, 1),
            "mean_error": error_estimates.squeeze(-1).float().mean().item()
                         if error_estimates.numel() > 0 else 0.0,
            "max_error": error_estimates.squeeze(-1).float().max().item()
                        if error_estimates.numel() > 0 else 0.0,
            "locked_slabs": self.locked.nonzero(as_tuple=True)[0].tolist(),
        }
        return stats
