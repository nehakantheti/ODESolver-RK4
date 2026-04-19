"""Parareal parallel-in-time ODE solver with neural coarse propagator.

Implements the Parareal algorithm that decomposes the time domain into
P parallel slabs, using a neural network as the fast coarse propagator
and classical RK4 as the accurate fine solver.

Algorithm overview:
    1. PARTITION: Split [t0, T] into P equal sub-intervals.
    2. INITIALISE: Run the NN coarse propagator sequentially to get
       initial guesses at every slab boundary.
    3. ITERATE until convergence:
       a. FINE PASS (parallel): Run RK4 on all active slabs (GPU-batched).
       b. TRUST GATE: Decide which slabs to correct vs skip.
       c. PARAREAL CORRECTION (sequential): Update slab boundaries using
          the correction formula:
          U_{n+1}^{k+1} = G(U_n^{k+1}) + [F(U_n^k) - G(U_n^k)]
       d. CONVERGENCE CHECK: max|U^{k+1} - U^k| < tolerance?

Speedup sources:
    - The fine pass is embarrassingly parallel (each slab is independent).
    - The coarse pass is a single NN forward pass (very fast on GPU).
    - The trust gate skips unnecessary fine corrections in later iterations.

Reference:
    - Lions, Maday, Turinici, "A parareal in time discretization of PDEs", 2001
    - Ibrahim et al., "Parareal with a PINN as coarse propagator", 2023
    - RandNet-Parareal, NeurIPS 2024
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from src.networks.coarse_propagator import CoarsePropagatorNet
from src.networks.trust_gate import TrustGate
from src.solvers.classical_rk4 import ClassicalRK4Solver, DerivativeFunc

logger = logging.getLogger(__name__)


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
    """Parareal parallel-in-time solver with neural coarse propagator.

    This solver combines a lightweight neural network (the coarse
    propagator) with a classical RK4 solver (the fine solver) in an
    iterative scheme that converges to the accuracy of the fine solver
    while exploiting time-parallelism.

    The key innovation is the use of a **meta-propagator** NN conditioned
    on ODE parameters, enabling the same coarse propagator to work across
    a family of related ODEs without retraining.

    Attributes:
        coarse_net: The neural coarse propagator network.
        fine_solver: Classical RK4 solver for the fine pass.
        trust_gate: Adaptive trust gate for selective fine correction.
        device: Torch device for computation.

    Example:
        >>> solver = PararealSolver(coarse_net=net, device=device)
        >>> result = solver.solve(
        ...     f=system.f, y0=y0, t_span=(0, 20),
        ...     n_slabs=16, fine_dt=0.001, params=params,
        ...     theta_ode=theta, tolerance=1e-6,
        ... )
    """

    def __init__(
        self,
        coarse_net: CoarsePropagatorNet,
        device: torch.device | None = None,
        trust_gate: TrustGate | None = None,
        max_iterations: int = 50,
    ):
        """Initialise the Parareal solver.

        Args:
            coarse_net: Trained neural coarse propagator.
            device: Torch device.  Defaults to CPU.
            trust_gate: Adaptive trust gate instance.  If ``None``, a
                       default gate is created.
            max_iterations: Maximum number of Parareal iterations before
                           giving up (prevents infinite loops).
        """
        self.coarse_net = coarse_net
        self.device = device or torch.device("cpu")
        self.fine_solver = ClassicalRK4Solver(device=self.device)
        self.trust_gate = trust_gate or TrustGate()
        self.max_iterations = max_iterations

        logger.info(
            "PararealSolver initialised: device=%s, max_iter=%d",
            self.device, max_iterations,
        )

    def _coarse_propagate(
        self,
        y_n: Tensor,
        t_n: float,
        delta_t: float,
        theta_ode: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Run the neural coarse propagator for one slab.

        Wraps ``coarse_net.predict()`` with logging.

        Args:
            y_n: Current state, shape ``(dim,)``.
            t_n: Current time.
            delta_t: Coarse time step.
            theta_ode: ODE parameter vector.

        Returns:
            Tuple of (predicted_state, confidence).
        """
        y_hat, confidence = self.coarse_net.predict(
            y_n, t_n, delta_t, theta_ode
        )
        return y_hat, confidence

    def _fine_solve_slab(
        self,
        f: DerivativeFunc,
        y_start: Tensor,
        t_start: float,
        t_end: float,
        dt: float,
        params: Dict[str, float],
    ) -> Tensor:
        """Run the fine RK4 solver over one time slab.

        Args:
            f: ODE derivative function.
            y_start: State at the start of the slab.
            t_start: Start time of the slab.
            t_end: End time of the slab.
            dt: Fine time step.
            params: ODE parameters.

        Returns:
            State at the end of the slab, shape ``(dim,)``.
        """
        return self.fine_solver.solve_interval(
            f=f, y0=y_start, t_start=t_start, t_end=t_end,
            dt=dt, params=params,
        )

    def solve(
        self,
        f: DerivativeFunc,
        y0: Tensor,
        t_span: Tuple[float, float],
        n_slabs: int,
        fine_dt: float,
        params: Dict[str, float],
        theta_ode: Tensor,
        tolerance: float = 1e-6,
        use_trust_gate: bool = True,
    ) -> PararealResult:
        """Solve an ODE using the Parareal algorithm.

        This is the main entry point.  It orchestrates the coarse-fine
        iteration loop until the solution converges to the accuracy of
        the fine solver (within ``tolerance``).

        How it works:
            1. Partition [t_start, t_end] into ``n_slabs`` equal intervals.
            2. Run the NN coarse propagator sequentially to get U_n^0.
            3. Iterate:
               a. Fine pass: solve each slab with RK4 (could be batched).
               b. Correction: apply the Parareal update formula.
               c. Check convergence.

        Args:
            f: ODE derivative function ``f(t, y, params)``.
            y0: Initial condition, shape ``(dim,)``.
            t_span: ``(t_start, t_end)``.
            n_slabs: Number of parallel time slabs (P).
            fine_dt: Step size for the fine RK4 solver.
            params: ODE parameter dictionary.
            theta_ode: Parameter vector for the coarse NN, shape
                      ``(param_dim,)``.
            tolerance: Convergence tolerance.
            use_trust_gate: Whether to use the adaptive trust gate.

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
        theta_ode = theta_ode.to(self.device, dtype=torch.float32)

        logger.info(
            "Parareal solve: n_slabs=%d, delta_T=%.4f, fine_dt=%.4f, "
            "tol=%.2e, trust_gate=%s",
            n_slabs, delta_t, fine_dt, tolerance, use_trust_gate,
        )

        # -- Step 1: Initial coarse pass (sequential) -----------------------
        U = torch.zeros(n_slabs + 1, dim, device=self.device)
        U[0] = y0
        confidences = torch.zeros(n_slabs, device=self.device)

        logger.info("Parareal: running initial coarse pass...")
        for n in range(n_slabs):
            y_hat, conf = self._coarse_propagate(
                U[n], slab_times[n].item(), delta_t, theta_ode
            )
            U[n + 1] = y_hat
            confidences[n] = conf.squeeze()

        # Cache G(U_n^k) for the correction formula
        G_old = torch.zeros(n_slabs, dim, device=self.device)
        for n in range(n_slabs):
            g_val, _ = self._coarse_propagate(
                U[n], slab_times[n].item(), delta_t, theta_ode
            )
            G_old[n] = g_val

        # -- Step 2: Iterative correction loop ------------------------------
        convergence_history = []
        fine_solves_list = []
        gate_stats_list = []

        self.trust_gate.reset()

        for k in range(self.max_iterations):
            U_old = U.clone()

            # Fine pass: solve each slab with RK4
            F_values = torch.zeros(n_slabs, dim, device=self.device)

            # Determine which slabs need fine correction
            if use_trust_gate and k > 0:
                error_estimates = 1.0 - confidences
                fine_mask = self.trust_gate.should_run_fine(error_estimates)
                gate_stats = self.trust_gate.get_stats(error_estimates)
                gate_stats_list.append(gate_stats)
                self.trust_gate.update_threshold(k)
            else:
                fine_mask = torch.ones(n_slabs, dtype=torch.bool,
                                       device=self.device)
                gate_stats_list.append({"trust_rate": 0.0})

            n_fine_this_iter = fine_mask.sum().item()
            fine_solves_list.append(int(n_fine_this_iter))

            logger.debug(
                "Parareal iter %d: running %d/%d fine solves",
                k, n_fine_this_iter, n_slabs,
            )

            # Run fine solver on active slabs (batched GPU execution).
            # All slab intervals have equal width delta_t.  For autonomous
            # ODEs (f does not depend on t), we can batch all ICs with a
            # shared t_span = (0, delta_t) and solve in one GPU kernel.
            active_idx = fine_mask.nonzero(as_tuple=True)[0]

            if len(active_idx) > 0:
                active_y0 = U_old[active_idx]
                endpoints = self.fine_solver.solve_batched_endpoints(
                    f=f,
                    y0_batch=active_y0,
                    t_span=(0.0, delta_t),
                    dt=fine_dt,
                    params=params,
                )
                F_values[active_idx] = endpoints

            # Fill skipped slabs with cached coarse predictions
            skipped_idx = (~fine_mask).nonzero(as_tuple=True)[0]
            if len(skipped_idx) > 0:
                F_values[skipped_idx] = G_old[skipped_idx]

            # Parareal correction (sequential due to dependency chain)
            for n in range(n_slabs):
                # New coarse prediction from updated initial value
                G_new, conf_new = self._coarse_propagate(
                    U[n], slab_times[n].item(), delta_t, theta_ode
                )
                confidences[n] = conf_new.squeeze()

                # Correction formula: G_new + (F_old - G_old)
                U[n + 1] = G_new + (F_values[n] - G_old[n])

                # Update cached G_old for next iteration
                G_old[n] = G_new

            # Convergence check
            max_change = torch.max(torch.abs(U - U_old)).item()
            convergence_history.append(max_change)

            logger.info(
                "Parareal iter %d: max_change=%.2e, fine_solves=%d/%d, "
                "trust_rate=%.1f%%",
                k, max_change, n_fine_this_iter, n_slabs,
                gate_stats_list[-1].get("trust_rate", 0) * 100,
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
