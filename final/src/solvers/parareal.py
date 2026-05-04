"""Parareal parallel-in-time ODE solver with pluggable coarse propagator.

Implements the Parareal algorithm that decomposes the time domain into
P parallel slabs, using a fast coarse propagator (neural or classical)
and classical RK4 as the accurate fine solver.

Coarse propagator options:
    1. **Neural** (default): Derivative-predicting NN, f̂(y, t, θ),
       integrated with Euler or RK2 over mini-steps.
    2. **Euler**: Forward Euler, y_{n+1} = y_n + dt * f(y_n, t_n).
    3. **Backward Euler**: Implicit, fixed-point iteration.

Algorithm overview:
    1. PARTITION: Split [t0, T] into P equal sub-intervals.
    2. INITIALISE: Run the coarse propagator sequentially to get
       initial guesses at every slab boundary.
    3. ITERATE until convergence:
       a. FINE PASS (parallel): Run RK4 on all active slabs.
       b. TRUST GATE: Lock converged slabs (skip fine on next iter).
       c. PARAREAL CORRECTION (sequential): Update slab boundaries:
          U_{n+1}^{k+1} = G(U_n^{k+1}) + [F(U_n^k) - G(U_n^k)]
       d. CONVERGENCE CHECK: max|U^{k+1} - U^k| < tolerance?

True parallelism:
    The fine pass uses multiprocessing.Pool with persistent pre-warmed
    workers for true multi-core CPU parallelism.  Each slab's RK4 runs
    on a separate core.

Reference:
    - Lions, Maday, Turinici, "A parareal in time discretization", 2001
    - RandNet-Parareal, NeurIPS 2024
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from src.networks.coarse_propagator import CoarsePropagatorNet
from src.networks.trust_gate import TrustGate
from src.solvers.classical_rk4 import ClassicalRK4Solver, DerivativeFunc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level worker for multiprocessing Pool (must be picklable)
# ---------------------------------------------------------------------------

# Global cache for worker processes — avoids re-importing per call
_worker_system = None
_worker_system_name = None


def _fine_worker_init(system_name: str):
    """Initialiser for pool workers — caches the ODE system.

    Called once when each worker process starts.  Avoids the overhead
    of re-importing and re-constructing the system on every task.

    Args:
        system_name: Registry name of the ODE system.
    """
    global _worker_system, _worker_system_name
    from src.ode_systems import get_system
    _worker_system = get_system(system_name)
    _worker_system_name = system_name


def _fine_solve_worker(args):
    """RK4 fine solve for one slab on CPU.

    Called by Pool.map().  Uses the cached ODE system from the
    worker initialiser (no per-call import overhead).

    Args:
        args: Tuple of (y0_list, t_start, t_end, dt, params).

    Returns:
        Final state as a Python list.
    """
    y0_list, t_start, t_end, dt, params = args

    global _worker_system
    import torch

    y = torch.tensor(y0_list, dtype=torch.float32)
    n_steps = int(round((t_end - t_start) / dt))
    f = _worker_system.f

    for i in range(n_steps):
        t = t_start + i * dt
        k1 = dt * f(t, y, params)
        k2 = dt * f(t + dt / 2, y + k1 / 2, params)
        k3 = dt * f(t + dt / 2, y + k2 / 2, params)
        k4 = dt * f(t + dt, y + k3, params)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y.tolist()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PararealResult:
    """Result from a Parareal solve.

    Attributes:
        t: Slab boundary times, shape ``(P + 1,)``.
        y: Solution at slab boundaries, shape ``(P + 1, dim)``.
        n_iterations: Number of Parareal iterations until convergence.
        convergence_history: Max error per iteration.
        fine_solves_per_iter: Number of fine solves triggered per iteration.
        trust_gate_stats: Per-iteration trust gate statistics.
        wall_time: Total wall-clock time in seconds.
    """
    t: Tensor
    y: Tensor
    n_iterations: int
    convergence_history: List[float] = field(default_factory=list)
    fine_solves_per_iter: List[int] = field(default_factory=list)
    trust_gate_stats: List[dict] = field(default_factory=list)
    wall_time: float = 0.0


# ---------------------------------------------------------------------------
# Parareal solver
# ---------------------------------------------------------------------------

class PararealSolver:
    """Parareal parallel-in-time solver with pluggable coarse propagator.

    Supports three coarse propagator modes:
        1. **neural**: Derivative-predicting NN (CoarsePropagatorNet)
        2. **euler**: Forward Euler (EulerCoarse)
        3. **backward_euler**: Implicit Backward Euler (BackwardEulerCoarse)

    The fine pass uses true multi-core parallelism via multiprocessing.Pool
    with persistent workers for minimal overhead.

    Attributes:
        coarse_net: Neural coarse propagator (if mode='neural').
        coarse_classical: Classical coarse propagator (if mode='euler'/'backward_euler').
        fine_solver: Classical RK4 solver for the fine pass.
        trust_gate: Convergence-based trust gate.
        device: Torch device for computation.

    Example:
        >>> solver = PararealSolver(coarse_net=net, device=device,
        ...                         system_name="damped_oscillator")
        >>> result = solver.solve(
        ...     f=system.f, y0=y0, t_span=(0, 20),
        ...     n_slabs=16, fine_dt=0.001, params=params,
        ...     theta_ode=theta, tolerance=1e-6,
        ... )
    """

    def __init__(
        self,
        coarse_net: CoarsePropagatorNet | None = None,
        device: torch.device | None = None,
        trust_gate: TrustGate | None = None,
        max_iterations: int = 50,
        coarse_dt: float = 0.1,
        n_workers: int = 0,
        system_name: str = "",
        coarse_mode: str = "neural",
        system=None,
    ):
        """Initialise the Parareal solver.

        Args:
            coarse_net: Trained neural coarse propagator (for mode='neural').
            device: Torch device for the coarse NN.  Defaults to CPU.
            trust_gate: Convergence-based trust gate.  If ``None``, a
                       default gate is created.
            max_iterations: Maximum Parareal iterations.
            coarse_dt: Step size for multi-step coarse propagation.
            n_workers: Number of CPU worker processes for fine pass.
                      0 = sequential CPU (vmap).
                      >0 = true multiprocessing parallelism.
            system_name: ODE system registry name.
            coarse_mode: 'neural', 'euler', or 'backward_euler'.
            system: ODE system instance (needed for classical coarse).
        """
        self.coarse_net = coarse_net
        self.device = device or torch.device("cpu")
        self.fine_solver = ClassicalRK4Solver(device=torch.device("cpu"))
        self.trust_gate = trust_gate or TrustGate()
        self.max_iterations = max_iterations
        self.coarse_dt = coarse_dt
        self.system_name = system_name
        self.n_workers = n_workers
        self.coarse_mode = coarse_mode
        self._pool = None

        # Set up classical coarse propagator if needed
        self.coarse_classical = None
        if coarse_mode in ("euler", "backward_euler", "rk2") and system is not None:
            from src.solvers.classical_coarse import (
                EulerCoarse, BackwardEulerCoarse, RK2Coarse,
            )
            if coarse_mode == "euler":
                self.coarse_classical = EulerCoarse(system, step_dt=coarse_dt)
            elif coarse_mode == "rk2":
                self.coarse_classical = RK2Coarse(system, step_dt=coarse_dt)
            else:
                self.coarse_classical = BackwardEulerCoarse(
                    system, step_dt=coarse_dt,
                )

        # Set up multiprocessing pool with persistent workers
        if n_workers > 0 and system_name:
            ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
            self._pool = ctx.Pool(
                processes=n_workers,
                initializer=_fine_worker_init,
                initargs=(system_name,),
            )
            logger.info(
                "Multiprocessing pool created: %d workers (method=%s)",
                n_workers, ctx.get_start_method(),
            )

        logger.info(
            "PararealSolver initialised: device=%s, max_iter=%d, "
            "coarse_dt=%.3f, coarse_mode=%s, n_workers=%d",
            self.device, max_iterations, coarse_dt, coarse_mode, n_workers,
        )

    def shutdown(self):
        """Shut down the process pool (if any)."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.shutdown()

    def _coarse_propagate(
        self,
        y_n: Tensor,
        t_n: float,
        delta_t: float,
        theta_ode: Tensor,
        params: Dict[str, float] | None = None,
    ) -> Tensor:
        """Run the coarse propagator across one slab.

        For neural mode: multi-step Euler integration using the learned
        derivative f̂ with steps of size coarse_dt.

        For classical mode: delegates to EulerCoarse or BackwardEulerCoarse.

        Args:
            y_n: Current state, shape ``(dim,)``.
            t_n: Current time.
            delta_t: Total time interval.
            theta_ode: ODE parameter vector (for neural mode).
            params: ODE parameter dict (for classical mode).

        Returns:
            Predicted state at t_n + delta_t, shape ``(dim,)``.
        """
        if self.coarse_mode == "neural" and self.coarse_net is not None:
            # Multi-step RK2 (Heun's method) with learned derivative f̂
            # RK2 is O(h²) vs Euler's O(h) — critical for reducing K.
            # Cost: 2 NN calls per step (vs 1 for Euler), still much
            # cheaper than fine RK4 (which does thousands of f evals).
            n_steps = max(1, round(delta_t / self.coarse_dt))
            step_dt = delta_t / n_steps
            y = y_n

            for i in range(n_steps):
                t_curr = t_n + i * step_dt
                # Stage 1: derivative at start
                k1 = self.coarse_net.predict(y, t_curr, theta_ode)
                # Stage 2: derivative at end (using Euler predictor)
                y_euler = y + step_dt * k1
                k2 = self.coarse_net.predict(y_euler, t_curr + step_dt, theta_ode)
                # Heun's update: average of start and end derivatives
                y = y + (step_dt / 2.0) * (k1 + k2)

            return y

        elif self.coarse_classical is not None:
            return self.coarse_classical.propagate(
                y_n, t_n, delta_t, params or {},
            )
        else:
            raise ValueError(
                f"No coarse propagator available for mode='{self.coarse_mode}'"
            )

    @torch.inference_mode()
    def solve(
        self,
        f: DerivativeFunc,
        y0: Tensor,
        t_span: Tuple[float, float],
        n_slabs: int,
        fine_dt: float,
        params: Dict[str, float],
        theta_ode: Tensor | None = None,
        tolerance: float = 1e-6,
        use_trust_gate: bool = True,
    ) -> PararealResult:
        """Solve an ODE using the Parareal algorithm.

        Args:
            f: ODE derivative function ``f(t, y, params)``.
            y0: Initial condition, shape ``(dim,)``.
            t_span: ``(t_start, t_end)``.
            n_slabs: Number of parallel time slabs (P).
            fine_dt: Step size for the fine RK4 solver.
            params: ODE parameter dictionary.
            theta_ode: Parameter vector for the coarse NN.
            tolerance: Convergence tolerance.
            use_trust_gate: Whether to use convergence-based slab locking.

        Returns:
            ``PararealResult`` with the converged solution and diagnostics.
        """
        start_time = time.time()
        t_start, t_end = t_span
        delta_t = (t_end - t_start) / n_slabs

        # Slab boundary times
        slab_times = torch.linspace(t_start, t_end, n_slabs + 1,
                                    device=self.device)

        dim = y0.shape[-1]
        y0 = y0.to(self.device, dtype=torch.float32)
        if theta_ode is not None:
            theta_ode = theta_ode.to(self.device, dtype=torch.float32)

        logger.info(
            "Parareal solve: n_slabs=%d, delta_T=%.4f, fine_dt=%.4f, "
            "tol=%.2e, coarse=%s, trust_gate=%s",
            n_slabs, delta_t, fine_dt, tolerance,
            self.coarse_mode, use_trust_gate,
        )

        # -- Step 1: Initial coarse pass (sequential) -----------------------
        U = torch.zeros(n_slabs + 1, dim, device=self.device)
        U[0] = y0

        logger.info("Parareal: running initial coarse pass...")
        for n in range(n_slabs):
            U[n + 1] = self._coarse_propagate(
                U[n], slab_times[n].item(), delta_t, theta_ode, params,
            )

        # Cache G(U_n^k) for the correction formula
        G_old = torch.zeros(n_slabs, dim, device=self.device)
        for n in range(n_slabs):
            G_old[n] = self._coarse_propagate(
                U[n], slab_times[n].item(), delta_t, theta_ode, params,
            )

        # -- Step 2: Iterative correction loop ------------------------------
        convergence_history = []
        fine_solves_list = []
        gate_stats_list = []

        # F_prev caches the fine-solve results from the previous iteration.
        # Needed for locked slabs: their F_values must persist, not be
        # replaced with G_old (which would zero out the correction).
        F_prev = G_old.clone()

        self.trust_gate.reset()

        for k in range(self.max_iterations):
            U_old = U.clone()

            # Fine pass: solve each slab with RK4
            F_values = torch.zeros(n_slabs, dim, device=self.device)

            # Determine which slabs need fine correction
            if use_trust_gate and k > 0:
                dummy_errors = torch.ones(n_slabs, device=self.device)
                fine_mask = self.trust_gate.should_run_fine(dummy_errors)
                gate_stats = self.trust_gate.get_stats(dummy_errors)
                gate_stats_list.append(gate_stats)
            else:
                fine_mask = torch.ones(n_slabs, dtype=torch.bool,
                                       device=self.device)
                gate_stats_list.append({"trust_rate": 0.0})

            n_fine_this_iter = fine_mask.sum().item()
            fine_solves_list.append(int(n_fine_this_iter))

            # Run fine solver on active slabs
            active_idx = fine_mask.nonzero(as_tuple=True)[0]

            if len(active_idx) > 0:
                if self._pool is not None and len(active_idx) > 1:
                    # ---- True parallel: multiprocessing.Pool ----
                    tasks = []
                    for idx in active_idx:
                        n_idx = idx.item()
                        y0_slab = U_old[n_idx].cpu().tolist()
                        t_slab_start = slab_times[n_idx].item()
                        tasks.append((
                            y0_slab,
                            t_slab_start,
                            t_slab_start + delta_t,
                            fine_dt,
                            params,
                        ))
                    results = self._pool.map(_fine_solve_worker, tasks)
                    for i, idx in enumerate(active_idx):
                        F_values[idx] = torch.tensor(
                            results[i], device=self.device,
                            dtype=torch.float32,
                        )
                else:
                    # ---- Sequential CPU: one-by-one fine solve ----
                    for idx in active_idx:
                        n_idx = idx.item()
                        y0_slab = U_old[n_idx].cpu()
                        t_slab_start = slab_times[n_idx].item()
                        endpoint = self.fine_solver.solve_interval(
                            f=f,
                            y0=y0_slab,
                            t_start=t_slab_start,
                            t_end=t_slab_start + delta_t,
                            dt=fine_dt,
                            params=params,
                        )
                        F_values[idx] = endpoint.to(self.device)

            # For skipped (locked) slabs, reuse PREVIOUS iteration's
            # fine result.  This preserves the correction formula:
            #   U_{n+1} = G_new + (F_old - G_old)
            # If we used G_old here, correction = G_new + 0 = G_new (wrong).
            skipped_idx = (~fine_mask).nonzero(as_tuple=True)[0]
            if len(skipped_idx) > 0:
                # F_prev was stored from last iteration
                F_values[skipped_idx] = F_prev[skipped_idx]

            # Parareal correction (sequential due to dependency chain)
            for n in range(n_slabs):
                G_new = self._coarse_propagate(
                    U[n], slab_times[n].item(), delta_t, theta_ode, params,
                )

                # Correction formula: G_new + (F_old - G_old)
                U[n + 1] = G_new + (F_values[n] - G_old[n])

                # Update cached G_old for next iteration
                G_old[n] = G_new

            # Save F_values for next iteration (locked slabs reuse these)
            F_prev = F_values.clone()

            # Convergence check
            max_change = torch.max(torch.abs(U - U_old)).item()
            convergence_history.append(max_change)

            # Compute per-slab changes and update trust gate locks
            if use_trust_gate:
                slab_changes = torch.zeros(n_slabs, device=self.device)
                for n in range(n_slabs):
                    slab_changes[n] = torch.max(
                        torch.abs(U[n + 1] - U_old[n + 1])
                    ).item()
                self.trust_gate.update_locks(slab_changes)

            n_locked = (self.trust_gate.locked.sum().item()
                        if self.trust_gate.locked is not None else 0)

            logger.info(
                "Parareal iter %d: max_change=%.2e, fine_solves=%d/%d, "
                "locked=%d/%d (trust=%.0f%%)",
                k, max_change, n_fine_this_iter, n_slabs,
                n_locked, n_slabs,
                n_locked / max(n_slabs, 1) * 100,
            )

            if max_change < tolerance:
                logger.info(
                    "Parareal CONVERGED at iteration %d (change=%.2e < tol=%.2e)",
                    k, max_change, tolerance,
                )
                break
        else:
            logger.warning(
                "Parareal did NOT converge after %d iterations "
                "(final change=%.2e, tol=%.2e)",
                self.max_iterations, max_change, tolerance,
            )

        wall_time = time.time() - start_time

        return PararealResult(
            t=slab_times,
            y=U,
            n_iterations=k + 1,
            convergence_history=convergence_history,
            fine_solves_per_iter=fine_solves_list,
            trust_gate_stats=gate_stats_list,
            wall_time=wall_time,
        )
