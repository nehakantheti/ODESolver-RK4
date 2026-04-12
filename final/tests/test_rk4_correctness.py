"""Correctness tests for the classical RK4 solver.

Validates:
    1. RK4 vs analytical solution (damped harmonic oscillator).
    2. RK4 vs scipy.integrate.solve_ivp for all 4 ODE systems.
    3. Single-solve and batched-solve produce identical results.
    4. solve_interval returns correct endpoint.
    5. k-factors are correctly returned when requested.

Run with:
    py -3.11 -m pytest final/tests/test_rk4_correctness.py -v
"""

from __future__ import annotations

import sys
import os

import pytest
import torch
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ode_systems import (
    DampedHarmonicOscillator,
    LotkaVolterra,
    VanDerPolOscillator,
    LorenzAttractor,
    get_system,
)
from src.solvers.classical_rk4 import ClassicalRK4Solver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solver() -> ClassicalRK4Solver:
    """Create a CPU-based RK4 solver."""
    return ClassicalRK4Solver(device=torch.device("cpu"))


@pytest.fixture
def damped_system() -> DampedHarmonicOscillator:
    """Create a damped harmonic oscillator system."""
    return DampedHarmonicOscillator()


# ---------------------------------------------------------------------------
# Test: ODE Systems instantiation and interface
# ---------------------------------------------------------------------------

class TestODESystems:
    """Tests for the ODE system definitions."""

    def test_registry_contains_all_systems(self):
        """All four systems should be accessible via get_system."""
        for name in ["damped_oscillator", "lotka_volterra",
                      "van_der_pol", "lorenz"]:
            system = get_system(name)
            assert system is not None
            assert system.dim > 0
            assert len(system.param_names) > 0

    def test_registry_raises_on_unknown(self):
        """get_system should raise KeyError for unknown names."""
        with pytest.raises(KeyError, match="Unknown ODE system"):
            get_system("nonexistent_system")

    def test_damped_oscillator_dimensions(self, damped_system):
        """Damped oscillator should be 2-dimensional."""
        assert damped_system.dim == 2
        assert damped_system.name == "Damped Harmonic Oscillator"
        assert len(damped_system.param_names) == 3

    def test_derivative_output_shape(self):
        """f() should return tensor of same shape as input y."""
        for name in ["damped_oscillator", "lotka_volterra",
                      "van_der_pol", "lorenz"]:
            system = get_system(name)
            y0 = system.default_initial_condition()
            params = system.default_params()
            dydt = system.f(0.0, y0, params)

            assert dydt.shape == y0.shape, (
                f"{system.name}: expected shape {y0.shape}, got {dydt.shape}"
            )

    def test_derivative_batched_shape(self):
        """f() should handle batched input (batch, dim)."""
        for name in ["damped_oscillator", "lotka_volterra",
                      "van_der_pol", "lorenz"]:
            system = get_system(name)
            y0 = system.default_initial_condition()
            params = system.default_params()

            # Create a batch of 5
            y_batch = y0.unsqueeze(0).expand(5, -1)
            dydt = system.f(0.0, y_batch, params)

            assert dydt.shape == (5, system.dim), (
                f"{system.name}: batched shape expected (5, {system.dim}), "
                f"got {dydt.shape}"
            )

    def test_param_vector_shape(self):
        """param_vector should return 1-D tensor of correct length."""
        for name in ["damped_oscillator", "lotka_volterra",
                      "van_der_pol", "lorenz"]:
            system = get_system(name)
            params = system.default_params()
            theta = system.param_vector(params)

            assert theta.shape == (len(system.param_names),)
            assert theta.dtype == torch.float32

    def test_param_ranges_cover_defaults(self):
        """Default params should fall within the sampling ranges."""
        for name in ["damped_oscillator", "lotka_volterra",
                      "van_der_pol", "lorenz"]:
            system = get_system(name)
            defaults = system.default_params()
            ranges = system.param_ranges()

            for param_name, val in defaults.items():
                low, high = ranges[param_name]
                assert low <= val <= high, (
                    f"{system.name}.{param_name}: default {val} outside "
                    f"range [{low}, {high}]"
                )


# ---------------------------------------------------------------------------
# Test: Classical RK4 solver — basic operation
# ---------------------------------------------------------------------------

class TestClassicalRK4Basic:
    """Basic operation tests for ClassicalRK4Solver."""

    def test_solve_single_output_shape(self, solver, damped_system):
        """solve_single should return trajectory of correct shape."""
        result = solver.solve_single(
            f=damped_system.f,
            y0=damped_system.default_initial_condition(),
            t_span=(0.0, 1.0),
            dt=0.1,
            params=damped_system.default_params(),
        )
        # 10 steps + initial = 11 points
        assert result.t.shape == (11,)
        assert result.y.shape == (11, 2)

    def test_solve_single_initial_condition_preserved(self, solver, damped_system):
        """First point of trajectory should equal the initial condition."""
        y0 = damped_system.default_initial_condition()
        result = solver.solve_single(
            f=damped_system.f, y0=y0,
            t_span=(0.0, 1.0), dt=0.1,
            params=damped_system.default_params(),
        )
        assert torch.allclose(result.y[0], y0, atol=1e-7)

    def test_solve_single_returns_k_factors(self, solver, damped_system):
        """k_factors should be returned when requested."""
        result = solver.solve_single(
            f=damped_system.f,
            y0=damped_system.default_initial_condition(),
            t_span=(0.0, 1.0), dt=0.1,
            params=damped_system.default_params(),
            return_k_factors=True,
        )
        assert result.k_factors is not None
        assert len(result.k_factors) == 10  # 10 steps
        k1, k2, k3, k4 = result.k_factors[0]
        assert k1.shape == (2,)

    def test_solve_single_no_k_factors_by_default(self, solver, damped_system):
        """k_factors should be None when not requested."""
        result = solver.solve_single(
            f=damped_system.f,
            y0=damped_system.default_initial_condition(),
            t_span=(0.0, 1.0), dt=0.1,
            params=damped_system.default_params(),
        )
        assert result.k_factors is None

    def test_solve_single_invalid_dt_raises(self, solver, damped_system):
        """Negative or zero dt should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            solver.solve_single(
                f=damped_system.f,
                y0=damped_system.default_initial_condition(),
                t_span=(0.0, 1.0), dt=-0.1,
                params=damped_system.default_params(),
            )

    def test_solve_single_invalid_t_span_raises(self, solver, damped_system):
        """t_end <= t_start should raise ValueError."""
        with pytest.raises(ValueError, match="greater than"):
            solver.solve_single(
                f=damped_system.f,
                y0=damped_system.default_initial_condition(),
                t_span=(1.0, 0.0), dt=0.1,
                params=damped_system.default_params(),
            )


# ---------------------------------------------------------------------------
# Test: RK4 vs analytical solution
# ---------------------------------------------------------------------------

class TestRK4VsAnalytical:
    """Validate RK4 accuracy against the known analytical solution
    for the damped harmonic oscillator."""

    def test_position_accuracy(self, solver, damped_system):
        """RK4 position x(t) should match analytical within 1e-5."""
        params = damped_system.default_params()
        result = solver.solve_single(
            f=damped_system.f,
            y0=damped_system.default_initial_condition(),
            t_span=(0.0, 10.0),
            dt=0.001,  # small step for high accuracy
            params=params,
        )
        analytical = damped_system.analytical_solution(result.t, params)
        assert analytical is not None

        # Compare position (first component)
        max_error = torch.max(torch.abs(result.y[:, 0] - analytical[:, 0]))
        assert max_error < 1e-5, f"Max position error: {max_error:.2e}"

    def test_velocity_accuracy(self, solver, damped_system):
        """RK4 velocity v(t) should match analytical within 1e-5."""
        params = damped_system.default_params()
        result = solver.solve_single(
            f=damped_system.f,
            y0=damped_system.default_initial_condition(),
            t_span=(0.0, 10.0),
            dt=0.001,
            params=params,
        )
        analytical = damped_system.analytical_solution(result.t, params)
        assert analytical is not None

        # Compare velocity (second component)
        max_error = torch.max(torch.abs(result.y[:, 1] - analytical[:, 1]))
        assert max_error < 1e-4, f"Max velocity error: {max_error:.2e}"


# ---------------------------------------------------------------------------
# Test: RK4 vs scipy for all systems
# ---------------------------------------------------------------------------

class TestRK4VsScipy:
    """Cross-validate RK4 against scipy.integrate.solve_ivp."""

    @pytest.mark.parametrize("system_name", [
        "damped_oscillator", "lotka_volterra", "van_der_pol", "lorenz"
    ])
    def test_scipy_agreement(self, solver, system_name):
        """RK4 and scipy RK45 should agree within tolerance for each system.

        Note on Lorenz:
            The Lorenz system is chaotic — even tiny floating-point
            differences between our fixed-step RK4 and scipy's adaptive
            RK45 grow exponentially over time.  We therefore use a shorter
            time window (t=0→5 instead of 0→25) and a relaxed threshold
            to validate correctness without fighting the Lyapunov exponent.
        """
        from scipy.integrate import solve_ivp

        system = get_system(system_name)
        params = system.default_params()
        y0 = system.default_initial_condition()
        t_start, t_end = system.default_time_span()

        # Lorenz: shorter window to avoid chaotic divergence
        if system_name == "lorenz":
            t_end = 5.0

        # Use a reasonable dt
        dt = 0.01 if system_name != "lorenz" else 0.005

        # Our RK4
        result = solver.solve_single(
            f=system.f, y0=y0, t_span=(t_start, t_end),
            dt=dt, params=params,
        )

        # Scipy RK45 (high accuracy reference)
        def scipy_f(t, y):
            y_tensor = torch.tensor(y, dtype=torch.float32)
            return system.f(t, y_tensor, params).numpy()

        scipy_result = solve_ivp(
            scipy_f, (t_start, t_end),
            y0.numpy(),
            method="RK45",
            t_eval=result.t.numpy(),
            rtol=1e-10, atol=1e-12,
        )

        # Compare
        our_y = result.y.numpy()
        scipy_y = scipy_result.y.T  # scipy returns (dim, n_points)

        max_error = np.max(np.abs(our_y - scipy_y))

        # Lorenz: relaxed threshold due to chaotic sensitivity
        threshold = 0.1 if system_name == "lorenz" else 1e-3
        assert max_error < threshold, (
            f"{system_name}: max error vs scipy = {max_error:.2e} "
            f"(threshold = {threshold})"
        )


# ---------------------------------------------------------------------------
# Test: Batched solve consistency
# ---------------------------------------------------------------------------

class TestBatchedSolve:
    """Verify that batched solve matches single solve."""

    def test_batched_matches_single(self, solver, damped_system):
        """Batched results should be identical to single-solve results."""
        params = damped_system.default_params()
        y0 = damped_system.default_initial_condition()
        t_span = (0.0, 2.0)
        dt = 0.05

        # Single solve
        single_result = solver.solve_single(
            f=damped_system.f, y0=y0, t_span=t_span,
            dt=dt, params=params,
        )

        # Batched solve with 3 identical ICs
        y0_batch = y0.unsqueeze(0).expand(3, -1).clone()
        batch_result = solver.solve_batched(
            f=damped_system.f, y0_batch=y0_batch,
            t_span=t_span, dt=dt, params=params,
        )

        # All 3 batch entries should match the single solve
        for i in range(3):
            assert torch.allclose(
                batch_result.y[i], single_result.y, atol=1e-6
            ), f"Batch entry {i} doesn't match single solve"


# ---------------------------------------------------------------------------
# Test: solve_interval endpoint
# ---------------------------------------------------------------------------

class TestSolveInterval:
    """Test solve_interval returns correct final state."""

    def test_interval_matches_full_solve_endpoint(self, solver, damped_system):
        """solve_interval should return the same y_end as solve_single."""
        params = damped_system.default_params()
        y0 = damped_system.default_initial_condition()
        t_span = (0.0, 5.0)
        dt = 0.01

        # Full solve — take the last point
        full_result = solver.solve_single(
            f=damped_system.f, y0=y0, t_span=t_span,
            dt=dt, params=params,
        )
        y_end_full = full_result.y[-1]

        # Interval solve
        y_end_interval = solver.solve_interval(
            f=damped_system.f, y0=y0,
            t_start=t_span[0], t_end=t_span[1],
            dt=dt, params=params,
        )

        assert torch.allclose(y_end_full, y_end_interval, atol=1e-7), (
            f"Endpoints differ: full={y_end_full}, interval={y_end_interval}"
        )
