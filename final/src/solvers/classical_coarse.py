"""Classical coarse propagators for Parareal baseline comparison.

Provides two classical alternatives to the neural coarse propagator:

1. **Euler**: y_{n+1} = y_n + dt * f(y_n, t_n)
   - Simplest, fastest, least accurate.
   - Expected: K ≈ P (bad coarse → large correction → slow convergence).

2. **Backward Euler**: y_{n+1} = y_n + dt * f(y_{n+1}, t_{n+1})
   - Implicit, unconditionally stable.
   - Solved via fixed-point iteration (3-5 iterations).
   - Expected: K ≈ P/2 (better stability → medium convergence).

Both implement the ``CoarsePropagator`` protocol so Parareal can swap
coarse solvers without changing any other code.

Usage:
    >>> from src.solvers.classical_coarse import EulerCoarse, BackwardEulerCoarse
    >>> coarse = EulerCoarse(system)
    >>> y_next = coarse.propagate(y_n, t_n=0.0, delta_t=1.0, params=params)
"""

from __future__ import annotations

import logging
from typing import Dict, Protocol, runtime_checkable

import torch
from torch import Tensor

from src.ode_systems import ODESystem

logger = logging.getLogger(__name__)


@runtime_checkable
class CoarsePropagator(Protocol):
    """Protocol for coarse propagators in the Parareal pipeline.

    Any coarse propagator (neural or classical) must implement this
    interface so PararealSolver can use it interchangeably.
    """

    def propagate(
        self,
        y_n: Tensor,
        t_n: float,
        delta_t: float,
        params: Dict[str, float],
    ) -> Tensor:
        """Propagate the ODE state from t_n to t_n + delta_t.

        Args:
            y_n: Current state, shape ``(dim,)``.
            t_n: Current time.
            delta_t: Time interval to propagate.
            params: ODE parameter dictionary.

        Returns:
            Predicted state at t_n + delta_t, shape ``(dim,)``.
        """
        ...


class EulerCoarse:
    """Forward Euler coarse propagator for Parareal baseline.

    Uses multi-step Euler integration with a configurable step size
    to propagate across a slab:

        for each step:
            y = y + dt * f(t, y, params)

    This is the simplest possible coarse propagator and serves as
    the lower-bound baseline for Parareal convergence.

    Attributes:
        system: The ODE system.
        step_dt: Step size for Euler integration within a slab.
    """

    def __init__(self, system: ODESystem, step_dt: float = 0.1):
        """Initialise the Euler coarse propagator.

        Args:
            system: ODE system instance.
            step_dt: Step size for multi-step Euler within each slab.
                    Smaller = more accurate but slower.
        """
        self.system = system
        self.step_dt = step_dt
        logger.info(
            "EulerCoarse created: system='%s', step_dt=%.3f",
            system.name, step_dt,
        )

    def propagate(
        self,
        y_n: Tensor,
        t_n: float,
        delta_t: float,
        params: Dict[str, float],
    ) -> Tensor:
        """Multi-step Euler propagation across a slab.

        Args:
            y_n: Current state, shape ``(dim,)``.
            t_n: Current time.
            delta_t: Total time interval.
            params: ODE parameter dictionary.

        Returns:
            Predicted state at t_n + delta_t.
        """
        n_steps = max(1, round(delta_t / self.step_dt))
        dt = delta_t / n_steps
        y = y_n.clone()

        for i in range(n_steps):
            t = t_n + i * dt
            y = y + dt * self.system.f(t, y, params)

        return y


class BackwardEulerCoarse:
    """Backward Euler (implicit) coarse propagator for Parareal baseline.

    Uses multi-step backward Euler with fixed-point iteration to solve
    the implicit equation:

        y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})

    The fixed-point iteration starts from a forward Euler prediction
    and converges in 3-5 iterations for typical stiffness.

    Attributes:
        system: The ODE system.
        step_dt: Step size for backward Euler within a slab.
        fp_iterations: Number of fixed-point iterations per step.
    """

    def __init__(
        self,
        system: ODESystem,
        step_dt: float = 0.1,
        fp_iterations: int = 5,
    ):
        """Initialise the Backward Euler coarse propagator.

        Args:
            system: ODE system instance.
            step_dt: Step size for multi-step backward Euler.
            fp_iterations: Number of fixed-point iterations for the
                          implicit solve.  3-5 is typically sufficient.
        """
        self.system = system
        self.step_dt = step_dt
        self.fp_iterations = fp_iterations
        logger.info(
            "BackwardEulerCoarse created: system='%s', step_dt=%.3f, "
            "fp_iters=%d",
            system.name, step_dt, fp_iterations,
        )

    def propagate(
        self,
        y_n: Tensor,
        t_n: float,
        delta_t: float,
        params: Dict[str, float],
    ) -> Tensor:
        """Multi-step backward Euler propagation across a slab.

        Args:
            y_n: Current state, shape ``(dim,)``.
            t_n: Current time.
            delta_t: Total time interval.
            params: ODE parameter dictionary.

        Returns:
            Predicted state at t_n + delta_t.
        """
        n_steps = max(1, round(delta_t / self.step_dt))
        dt = delta_t / n_steps
        y = y_n.clone()

        for i in range(n_steps):
            t_next = t_n + (i + 1) * dt

            # Initial guess: forward Euler
            y_guess = y + dt * self.system.f(t_n + i * dt, y, params)

            # Fixed-point iteration for implicit solve
            for _ in range(self.fp_iterations):
                y_guess = y + dt * self.system.f(t_next, y_guess, params)

            y = y_guess

        return y
