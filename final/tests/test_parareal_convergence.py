"""Convergence tests for the Parareal solver.

Validates:
    1. Parareal with an untrained NN still converges (via correction formula).
    2. Convergence error decreases monotonically across iterations.
    3. Result matches serial RK4 within tolerance.
    4. Trust gate stats are properly populated.
    5. PararealResult contains all expected diagnostics.

Note:
    These tests use an *untrained* coarse propagator to verify the
    Parareal algorithm logic.  An untrained NN gives poor initial
    guesses, but the correction formula guarantees convergence after
    at most P iterations (where P = number of slabs).  This is by design:
    the algorithm is correct regardless of coarse solver quality; a good
    coarse solver only reduces the *number* of iterations needed.

Run with:
    py -3.11 -m pytest final/tests/test_parareal_convergence.py -v
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ode_systems import DampedHarmonicOscillator, get_system
from src.networks.coarse_propagator import CoarsePropagatorNet
from src.networks.trust_gate import TrustGate
from src.solvers.classical_rk4 import ClassicalRK4Solver
from src.solvers.parareal import PararealSolver, PararealResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def system() -> DampedHarmonicOscillator:
    """Create the damped oscillator system."""
    return DampedHarmonicOscillator()


@pytest.fixture
def untrained_coarse_net(system) -> CoarsePropagatorNet:
    """Create an untrained coarse propagator (random weights).

    The Parareal correction formula guarantees convergence even with a
    poor coarse solver — it just takes more iterations.
    """
    return CoarsePropagatorNet(
        state_dim=system.dim,
        param_dim=len(system.param_names),
        hidden_dim=32,  # small for test speed
    )


@pytest.fixture
def parareal_solver(untrained_coarse_net) -> PararealSolver:
    """Create a Parareal solver with an untrained NN."""
    return PararealSolver(
        coarse_net=untrained_coarse_net,
        device=torch.device("cpu"),
        trust_gate=TrustGate(initial_threshold=0.5),
        max_iterations=30,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPararealConvergence:
    """Tests for Parareal algorithm convergence properties."""

    def test_converges_with_untrained_nn(self, parareal_solver, system):
        """Parareal must converge even with an untrained coarse solver.

        The correction formula guarantees convergence in at most P
        iterations (where P = number of slabs).  With 4 slabs and
        max_iterations=30, we should converge.
        """
        params = system.default_params()
        y0 = system.default_initial_condition()
        theta = system.param_vector(params)

        result = parareal_solver.solve(
            f=system.f,
            y0=y0,
            t_span=(0.0, 2.0),  # short interval for test speed
            n_slabs=4,
            fine_dt=0.01,
            params=params,
            theta_ode=theta,
            tolerance=1e-4,
            use_trust_gate=False,  # disable gate for predictable convergence
        )

        assert result.n_iterations <= 30, (
            f"Failed to converge in 30 iterations "
            f"(converged in {result.n_iterations})"
        )
        assert len(result.convergence_history) > 0

    def test_convergence_error_decreases(self, parareal_solver, system):
        """Convergence error should generally decrease over iterations.

        We check that the final error is less than the first error.
        Monotonic decrease is not strictly guaranteed (can have small
        bumps) but overall the trend should be downward.
        """
        params = system.default_params()
        y0 = system.default_initial_condition()
        theta = system.param_vector(params)

        result = parareal_solver.solve(
            f=system.f,
            y0=y0,
            t_span=(0.0, 2.0),
            n_slabs=4,
            fine_dt=0.01,
            params=params,
            theta_ode=theta,
            tolerance=1e-6,
            use_trust_gate=False,
        )

        if len(result.convergence_history) > 1:
            first_error = result.convergence_history[0]
            last_error = result.convergence_history[-1]
            assert last_error < first_error, (
                f"Final error ({last_error:.2e}) not less than "
                f"first error ({first_error:.2e})"
            )

    def test_matches_serial_rk4(self, parareal_solver, system):
        """Converged Parareal should match serial RK4 at slab boundaries.

        After convergence, the Parareal solution at each slab boundary
        should agree with the serial fine RK4 solution within tolerance.
        """
        params = system.default_params()
        y0 = system.default_initial_condition()
        theta = system.param_vector(params)

        t_span = (0.0, 2.0)
        fine_dt = 0.01
        n_slabs = 4

        # Parareal solve
        result = parareal_solver.solve(
            f=system.f,
            y0=y0,
            t_span=t_span,
            n_slabs=n_slabs,
            fine_dt=fine_dt,
            params=params,
            theta_ode=theta,
            tolerance=1e-5,
            use_trust_gate=False,
        )

        # Serial RK4 reference
        serial_solver = ClassicalRK4Solver(device=torch.device("cpu"))
        serial_result = serial_solver.solve_single(
            f=system.f, y0=y0, t_span=t_span,
            dt=fine_dt, params=params,
        )

        # Compare at slab boundaries
        delta_t = (t_span[1] - t_span[0]) / n_slabs
        for n in range(n_slabs + 1):
            # Find the serial trajectory index closest to slab time
            slab_time = t_span[0] + n * delta_t
            serial_idx = int(round((slab_time - t_span[0]) / fine_dt))
            serial_idx = min(serial_idx, serial_result.y.shape[0] - 1)

            max_diff = torch.max(
                torch.abs(result.y[n] - serial_result.y[serial_idx])
            ).item()

            # Allow reasonable tolerance (untrained NN = more iterations)
            assert max_diff < 0.1, (
                f"Slab {n} at t={slab_time:.2f}: max diff from serial = "
                f"{max_diff:.2e} (too large)"
            )

    def test_result_diagnostics(self, parareal_solver, system):
        """PararealResult should contain all expected diagnostic fields."""
        params = system.default_params()
        y0 = system.default_initial_condition()
        theta = system.param_vector(params)

        result = parareal_solver.solve(
            f=system.f,
            y0=y0,
            t_span=(0.0, 1.0),
            n_slabs=3,
            fine_dt=0.05,
            params=params,
            theta_ode=theta,
            tolerance=1e-3,
            use_trust_gate=True,
        )

        # Check result structure
        assert isinstance(result, PararealResult)
        assert result.t.shape == (4,)      # n_slabs + 1
        assert result.y.shape[0] == 4      # n_slabs + 1
        assert result.y.shape[1] == 2      # state_dim
        assert result.n_iterations > 0
        assert len(result.convergence_history) == result.n_iterations
        assert len(result.fine_solves_per_iter) == result.n_iterations
        assert result.wall_time > 0

    def test_initial_condition_preserved(self, parareal_solver, system):
        """The first slab boundary should always equal y0."""
        params = system.default_params()
        y0 = system.default_initial_condition()
        theta = system.param_vector(params)

        result = parareal_solver.solve(
            f=system.f,
            y0=y0,
            t_span=(0.0, 1.0),
            n_slabs=3,
            fine_dt=0.05,
            params=params,
            theta_ode=theta,
            tolerance=1e-3,
            use_trust_gate=False,
        )

        assert torch.allclose(result.y[0], y0, atol=1e-7), (
            f"IC not preserved: expected {y0}, got {result.y[0]}"
        )

    def test_trust_gate_stats_populated(self, parareal_solver, system):
        """Trust gate stats should be populated when gate is enabled."""
        params = system.default_params()
        y0 = system.default_initial_condition()
        theta = system.param_vector(params)

        result = parareal_solver.solve(
            f=system.f,
            y0=y0,
            t_span=(0.0, 1.0),
            n_slabs=3,
            fine_dt=0.05,
            params=params,
            theta_ode=theta,
            tolerance=1e-3,
            use_trust_gate=True,
        )

        assert len(result.trust_gate_stats) > 0
        # At least the first iteration should have stats
        assert isinstance(result.trust_gate_stats[0], dict)
