"""Classical Runge-Kutta 4th order (RK4) ODE solver in pure PyTorch.

Provides three execution modes with identical numerical behaviour:

1. ``solve_single``  — One trajectory, one set of initial conditions.
2. ``solve_batched`` — Many trajectories in parallel via ``torch.vmap``
                       (used by the Parareal fine pass).
3. ``solve_gpu``     — ``solve_batched`` wrapped with AMP and
                       ``torch.compile`` for maximum GPU throughput.

Design decisions:
    * The solver is a *class* (``ClassicalRK4Solver``) so it can be
      swapped with neural-augmented variants via the same interface
      (Open/Closed Principle — SOLID).
    * The ``f`` callback follows the signature ``f(t, y, params)`` to
      match ``ODESystem.f``, keeping coupling loose.
    * All intermediate k-factors are returned when ``return_k_factors``
      is set, providing training data for the k-factor residual network.

Mathematical formulation (standard RK4):
    k₁ = h · f(tₙ, yₙ)
    k₂ = h · f(tₙ + h/2, yₙ + k₁/2)
    k₃ = h · f(tₙ + h/2, yₙ + k₂/2)
    k₄ = h · f(tₙ + h,   yₙ + k₃)
    yₙ₊₁ = yₙ + (1/6)(k₁ + 2k₂ + 2k₃ + k₄)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RK4StepResult:
    """Result of a single RK4 step, optionally including k-factors.

    Attributes:
        y_next: State after one step, shape ``(dim,)`` or ``(batch, dim)``.
        k1: First slope evaluation, same shape as ``y_next``.
        k2: Second slope evaluation.
        k3: Third slope evaluation.
        k4: Fourth slope evaluation.
    """
    y_next: Tensor
    k1: Tensor
    k2: Tensor
    k3: Tensor
    k4: Tensor


@dataclass
class SolveResult:
    """Full trajectory returned by the solver.

    Attributes:
        t: 1-D tensor of time points, shape ``(n_steps + 1,)``.
        y: State trajectory, shape ``(n_steps + 1, dim)`` (single) or
           ``(batch, n_steps + 1, dim)`` (batched).
        k_factors: If requested, list of ``(k1, k2, k3, k4)`` tuples for
                   each step.  ``None`` otherwise.
    """
    t: Tensor
    y: Tensor
    k_factors: Optional[List[Tuple[Tensor, Tensor, Tensor, Tensor]]] = None


# ---------------------------------------------------------------------------
# Derivative function type alias
# ---------------------------------------------------------------------------

# f(t, y, params) -> dy/dt
DerivativeFunc = Callable[[float, Tensor, Dict[str, float]], Tensor]


# ---------------------------------------------------------------------------
# RK4 Solver
# ---------------------------------------------------------------------------

class ClassicalRK4Solver:
    """Classical 4th-order Runge-Kutta ODE solver (explicit, fixed step).

    This solver implements the standard RK4 method with a fixed time step.
    It supports single and batched (vmap-parallelised) execution modes.

    Attributes:
        device: Torch device to run computations on.

    Example:
        >>> from src.ode_systems import DampedHarmonicOscillator
        >>> system = DampedHarmonicOscillator()
        >>> solver = ClassicalRK4Solver(device=torch.device("cpu"))
        >>> result = solver.solve_single(
        ...     f=system.f,
        ...     y0=system.default_initial_condition(),
        ...     t_span=system.default_time_span(),
        ...     dt=0.01,
        ...     params=system.default_params(),
        ... )
        >>> result.y.shape  # (n_steps+1, 2)
    """

    def __init__(self, device: torch.device | None = None):
        """Initialise the solver.

        Args:
            device: Torch device (``"cpu"`` or ``"cuda"``).  Defaults to
                    CPU if not specified.
        """
        self.device = device or torch.device("cpu")
        logger.info("ClassicalRK4Solver initialised on device=%s", self.device)

    # -- Single-step helper --------------------------------------------------

    @staticmethod
    def rk4_step(
        f: DerivativeFunc,
        t: float,
        y: Tensor,
        h: float,
        params: Dict[str, float],
    ) -> RK4StepResult:
        """Perform one RK4 step from ``(t, y)`` with step size ``h``.

        Computes the four slope evaluations k₁–k₄ and returns the updated
        state along with all k-factors (for downstream training data).

        Args:
            f: Derivative function ``f(t, y, params) -> dy/dt``.
            t: Current time.
            y: Current state, shape ``(dim,)`` or ``(batch, dim)``.
            h: Step size.
            params: ODE parameter dictionary.

        Returns:
            ``RK4StepResult`` containing the next state and all 4 k-factors.
        """
        k1 = h * f(t, y, params)
        k2 = h * f(t + h / 2.0, y + k1 / 2.0, params)
        k3 = h * f(t + h / 2.0, y + k2 / 2.0, params)
        k4 = h * f(t + h, y + k3, params)

        y_next = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        return RK4StepResult(y_next=y_next, k1=k1, k2=k2, k3=k3, k4=k4)

    # -- Full trajectory (single) -------------------------------------------

    def solve_single(
        self,
        f: DerivativeFunc,
        y0: Tensor,
        t_span: Tuple[float, float],
        dt: float,
        params: Dict[str, float],
        return_k_factors: bool = False,
    ) -> SolveResult:
        """Integrate one trajectory using RK4 with fixed step size.

        This is the straightforward sequential solver.  It steps through
        time from ``t_span[0]`` to ``t_span[1]`` in increments of ``dt``,
        recording the state at every step.

        Args:
            f: Derivative function from an ``ODESystem``.
            y0: Initial condition, shape ``(dim,)``.
            t_span: ``(t_start, t_end)``.
            dt: Fixed time step.
            params: ODE parameter dictionary.
            return_k_factors: If ``True``, also store all intermediate
                              k-factors (needed for training data generation).

        Returns:
            ``SolveResult`` with time points and state trajectory.

        Raises:
            ValueError: If ``dt <= 0`` or ``t_span`` is invalid.
        """
        t_start, t_end = t_span
        if dt <= 0:
            raise ValueError(f"Step size dt must be positive, got {dt}")
        if t_end <= t_start:
            raise ValueError(
                f"t_end must be greater than t_start, got ({t_start}, {t_end})"
            )

        # Move initial condition to the correct device
        y = y0.to(self.device, dtype=torch.float32)

        # Build the time grid
        n_steps = int((t_end - t_start) / dt)
        t_points = torch.linspace(t_start, t_start + n_steps * dt,
                                  n_steps + 1, device=self.device)

        logger.debug(
            "solve_single: t=[%.2f, %.2f], dt=%.4f, n_steps=%d",
            t_start, t_end, dt, n_steps,
        )

        # Pre-allocate output trajectory
        trajectory = torch.zeros(n_steps + 1, y.shape[-1],
                                 device=self.device, dtype=torch.float32)
        trajectory[0] = y

        k_factors_list: Optional[List[Tuple[Tensor, ...]]] = (
            [] if return_k_factors else None
        )

        # Time-stepping loop
        for i in range(n_steps):
            t_current = t_points[i].item()
            step = self.rk4_step(f, t_current, y, dt, params)
            y = step.y_next
            trajectory[i + 1] = y

            if k_factors_list is not None:
                k_factors_list.append((step.k1, step.k2, step.k3, step.k4))

        logger.debug("solve_single: completed %d steps", n_steps)
        return SolveResult(t=t_points, y=trajectory, k_factors=k_factors_list)

    # -- Full trajectory (batched via vmap) ----------------------------------

    def solve_batched(
        self,
        f: DerivativeFunc,
        y0_batch: Tensor,
        t_span: Tuple[float, float],
        dt: float,
        params: Dict[str, float],
    ) -> SolveResult:
        """Integrate multiple trajectories in parallel using ``torch.vmap``.

        Each row of ``y0_batch`` is a different initial condition.  All
        trajectories share the same time grid and ODE parameters.

        This is the core of the Parareal fine-pass parallelism: each
        time slab starts from a different IC (provided by the coarse
        propagator) and runs RK4 independently — all in one GPU kernel.

        How it works:
            1. Define a single-trajectory solver as a pure function.
            2. Use ``torch.vmap`` to vectorise over the batch dimension.
            3. Execute — PyTorch fuses the batch into a single kernel.

        Args:
            f: Derivative function from an ``ODESystem``.
            y0_batch: Batch of initial conditions, shape ``(batch, dim)``.
            t_span: ``(t_start, t_end)`` — same for all trajectories.
            dt: Fixed time step.
            params: ODE parameter dictionary (shared across the batch).

        Returns:
            ``SolveResult`` with ``y`` of shape ``(batch, n_steps+1, dim)``.
        """
        batch_size = y0_batch.shape[0]
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)
        t_points = torch.linspace(t_start, t_start + n_steps * dt,
                                  n_steps + 1, device=self.device)

        logger.info(
            "solve_batched: batch_size=%d, n_steps=%d, device=%s",
            batch_size, n_steps, self.device,
        )

        # Move batch to device
        y0_batch = y0_batch.to(self.device, dtype=torch.float32)

        def _solve_one(y0_single: Tensor) -> Tensor:
            """Solve a single trajectory (pure function for vmap).

            Args:
                y0_single: Initial condition, shape ``(dim,)``.

            Returns:
                Full trajectory, shape ``(n_steps + 1, dim)``.
            """
            y = y0_single
            dim = y.shape[-1]
            traj = torch.zeros(n_steps + 1, dim,
                               device=y.device, dtype=y.dtype)
            traj[0] = y

            for i in range(n_steps):
                t_curr = t_start + i * dt
                k1 = dt * f(t_curr, y, params)
                k2 = dt * f(t_curr + dt / 2.0, y + k1 / 2.0, params)
                k3 = dt * f(t_curr + dt / 2.0, y + k2 / 2.0, params)
                k4 = dt * f(t_curr + dt, y + k3, params)
                y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                traj[i + 1] = y

            return traj

        # Attempt vmap; fall back to manual loop if unsupported
        try:
            batched_solve = torch.vmap(_solve_one)
            trajectories = batched_solve(y0_batch)
            logger.debug("solve_batched: vmap execution succeeded")
        except RuntimeError as exc:
            logger.warning(
                "vmap failed (%s), falling back to sequential loop", exc
            )
            trajectories = torch.stack(
                [_solve_one(y0_batch[i]) for i in range(batch_size)]
            )

        return SolveResult(t=t_points, y=trajectories)

    # -- Batched endpoint-only solver (for Parareal fine pass) ---------------

    def solve_batched_endpoints(
        self,
        f: DerivativeFunc,
        y0_batch: Tensor,
        t_span: Tuple[float, float],
        dt: float,
        params: Dict[str, float],
    ) -> Tensor:
        """Integrate multiple trajectories in parallel, returning only endpoints.

        Optimised for the Parareal fine pass: runs *P* slab-level RK4
        solves simultaneously using ``torch.vmap``, returning only the
        final state of each trajectory (no trajectory storage ⇒ less VRAM).

        How it works:
            1. Define a pure-function endpoint solver (no trajectory storage).
            2. Vectorise it with ``torch.vmap`` over the batch of ICs.
            3. All P fine solves execute in a single fused GPU kernel.

        When ``vmap`` is incompatible with the derivative function ``f``
        (e.g. Python-level control flow inside ``f``), the method
        automatically falls back to a sequential loop.

        Args:
            f: Derivative function from an ``ODESystem``.
            y0_batch: Batch of initial conditions, shape ``(batch, dim)``.
            t_span: ``(t_start, t_end)`` — same interval for all trajectories.
                   For autonomous ODEs the absolute time doesn't matter.
            dt: Fixed time step.
            params: ODE parameter dictionary (shared across the batch).

        Returns:
            Final states tensor of shape ``(batch, dim)``.
        """
        batch_size = y0_batch.shape[0]
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        # Move batch to device
        y0_batch = y0_batch.to(self.device, dtype=torch.float32)

        logger.debug(
            "solve_batched_endpoints: batch=%d, n_steps=%d, device=%s",
            batch_size, n_steps, self.device,
        )

        def _solve_one_endpoint(y0_single: Tensor) -> Tensor:
            """Solve one trajectory, return only the final state.

            This is a pure function (no side-effects, no trajectory
            storage) so ``torch.vmap`` can vectorise it.

            Args:
                y0_single: Initial condition, shape ``(dim,)``.

            Returns:
                Final state, shape ``(dim,)``.
            """
            y = y0_single
            for i in range(n_steps):
                t_curr = t_start + i * dt
                k1 = dt * f(t_curr, y, params)
                k2 = dt * f(t_curr + dt / 2.0, y + k1 / 2.0, params)
                k3 = dt * f(t_curr + dt / 2.0, y + k2 / 2.0, params)
                k4 = dt * f(t_curr + dt, y + k3, params)
                y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            return y

        # Attempt vmap; fall back to sequential loop if unsupported
        try:
            batched_solve = torch.vmap(_solve_one_endpoint)
            endpoints = batched_solve(y0_batch)
            logger.debug("solve_batched_endpoints: vmap execution succeeded")
        except RuntimeError as exc:
            logger.warning(
                "vmap failed (%s), falling back to sequential loop", exc,
            )
            endpoints = torch.stack(
                [_solve_one_endpoint(y0_batch[i]) for i in range(batch_size)]
            )

        return endpoints

    def solve_interval(
        self,
        f: DerivativeFunc,
        y0: Tensor,
        t_start: float,
        t_end: float,
        dt: float,
        params: Dict[str, float],
    ) -> Tensor:
        """Integrate from ``t_start`` to ``t_end`` and return only the
        final state (no trajectory storage).

        This is optimised for the Parareal fine pass where we only need
        the endpoint, not the full history.

        Args:
            f: Derivative function from an ``ODESystem``.
            y0: Initial condition, shape ``(dim,)``.
            t_start: Start of the sub-interval.
            t_end: End of the sub-interval.
            dt: Fixed time step.
            params: ODE parameter dictionary.

        Returns:
            Final state tensor of shape ``(dim,)``.
        """
        n_steps = int((t_end - t_start) / dt)
        y = y0.to(self.device, dtype=torch.float32)

        for i in range(n_steps):
            t_curr = t_start + i * dt
            step = self.rk4_step(f, t_curr, y, dt, params)
            y = step.y_next

        return y
