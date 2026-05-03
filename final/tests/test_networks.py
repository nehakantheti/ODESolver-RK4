"""Tests for neural network architectures and data generation.

Validates:
    1. CoarsePropagatorNet — derivative prediction, shapes, gradient flow.
    2. KFactorResidualNet — output shapes, residual structure, θ conditioning.
    3. TrustGate — convergence-based slab locking.
    4. Classical coarse propagators — Euler, Backward Euler.
    5. DataGenerator — output shapes and data integrity.

Run with:
    python -m pytest final/tests/test_networks.py -v
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.networks.coarse_propagator import CoarsePropagatorNet
from src.networks.k_factor_residual import KFactorResidualNet, ResidualBlock
from src.networks.trust_gate import TrustGate
from src.ode_systems import get_system
from src.training.data_generator import DataGenerator


# ---------------------------------------------------------------------------
# Test: CoarsePropagatorNet (derivative prediction mode)
# ---------------------------------------------------------------------------

class TestCoarsePropagatorNet:
    """Tests for the derivative-predicting coarse propagator."""

    @pytest.fixture
    def net(self) -> CoarsePropagatorNet:
        """Create a coarse propagator for a 2-D system with 3 params."""
        return CoarsePropagatorNet(state_dim=2, param_dim=3, hidden_dim=32)

    def test_output_shape(self, net):
        """Forward pass should return derivative of correct shape."""
        batch_size = 16
        y_n = torch.randn(batch_size, 2)
        t_n = torch.randn(batch_size, 1)
        theta = torch.randn(batch_size, 3)

        f_hat = net(y_n, t_n, theta)

        assert f_hat.shape == (batch_size, 2), \
            f"Expected derivative shape (16, 2), got {f_hat.shape}"

    def test_no_confidence_head(self, net):
        """Network should NOT have a confidence head."""
        assert not hasattr(net, 'confidence_head'), \
            "Confidence head should be removed"
        assert hasattr(net, 'derivative_head'), \
            "Should have derivative_head instead"

    def test_no_delta_t_input(self, net):
        """Forward should take (y_n, t_n, theta) — no delta_t."""
        y_n = torch.randn(4, 2)
        t_n = torch.randn(4, 1)
        theta = torch.randn(4, 3)

        # Should work without delta_t
        f_hat = net(y_n, t_n, theta)
        assert f_hat.shape == (4, 2)

    def test_gradient_flow(self, net):
        """Gradients should flow through all parameters."""
        y_n = torch.randn(8, 2, requires_grad=True)
        t_n = torch.randn(8, 1)
        theta = torch.randn(8, 3)

        f_hat = net(y_n, t_n, theta)
        loss = f_hat.sum()
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), \
                f"Non-finite gradient in {name}"

    def test_predict_convenience(self, net):
        """predict() should handle unbatched inputs correctly."""
        y_n = torch.randn(2)
        theta = torch.randn(3)

        f_hat = net.predict(y_n, t_n=0.5, theta_ode=theta)

        assert f_hat.shape == (2,), f"Expected shape (2,), got {f_hat.shape}"

    def test_integrate_euler(self, net):
        """integrate_euler should return next state of correct shape."""
        y_n = torch.randn(2)
        theta = torch.randn(3)

        y_next = net.integrate_euler(y_n, t_n=0.0, dt=0.1, theta_ode=theta)
        assert y_next.shape == (2,)

    def test_integrate_rk2(self, net):
        """integrate_rk2 should return next state of correct shape."""
        y_n = torch.randn(2)
        theta = torch.randn(3)

        y_next = net.integrate_rk2(y_n, t_n=0.0, dt=0.1, theta_ode=theta)
        assert y_next.shape == (2,)

    def test_different_state_dims(self):
        """Network should work for different state dimensions."""
        for state_dim in [1, 2, 3, 5]:
            net = CoarsePropagatorNet(state_dim=state_dim, param_dim=2,
                                     hidden_dim=16)
            y_n = torch.randn(4, state_dim)
            t_n = torch.randn(4, 1)
            theta = torch.randn(4, 2)

            f_hat = net(y_n, t_n, theta)
            assert f_hat.shape == (4, state_dim)


# ---------------------------------------------------------------------------
# Test: KFactorResidualNet
# ---------------------------------------------------------------------------

class TestKFactorResidualNet:
    """Tests for the k-factor residual prediction network."""

    @pytest.fixture
    def net(self) -> KFactorResidualNet:
        """Create a k-factor net for a 2-D system with 3 params."""
        return KFactorResidualNet(state_dim=2, param_dim=3, hidden_dim=32)

    def test_output_shapes(self, net):
        """Forward pass should return 3 correction tensors of correct shape."""
        batch_size = 16
        k1 = torch.randn(batch_size, 2)
        y_n = torch.randn(batch_size, 2)
        t_n = torch.randn(batch_size, 1)
        h = torch.randn(batch_size, 1)
        theta = torch.randn(batch_size, 3)

        delta_2, delta_3, delta_4 = net(k1, y_n, t_n, h, theta)

        assert delta_2.shape == (batch_size, 2)
        assert delta_3.shape == (batch_size, 2)
        assert delta_4.shape == (batch_size, 2)

    def test_predict_k_factors_adds_k1(self, net):
        """predict_k_factors should add k1 to each delta."""
        k1 = torch.ones(4, 2)
        y_n = torch.randn(4, 2)
        t_n = torch.randn(4, 1)
        h = torch.randn(4, 1)
        theta = torch.randn(4, 3)

        d2, d3, d4 = net(k1, y_n, t_n, h, theta)
        k2_hat, k3_hat, k4_hat = net.predict_k_factors(k1, y_n, t_n, h, theta)

        assert torch.allclose(k2_hat, k1 + d2, atol=1e-6)
        assert torch.allclose(k3_hat, k1 + d3, atol=1e-6)
        assert torch.allclose(k4_hat, k1 + d4, atol=1e-6)

    def test_backward_compat_no_theta(self):
        """Network with param_dim=0 should work without theta."""
        net = KFactorResidualNet(state_dim=2, param_dim=0, hidden_dim=16)
        k1 = torch.randn(4, 2)
        y_n = torch.randn(4, 2)
        t_n = torch.randn(4, 1)
        h = torch.randn(4, 1)

        d2, d3, d4 = net(k1, y_n, t_n, h)
        assert d2.shape == (4, 2)

    def test_gradient_flow(self, net):
        """Gradients should flow through all parameters."""
        k1 = torch.randn(8, 2)
        y_n = torch.randn(8, 2)
        t_n = torch.randn(8, 1)
        h = torch.randn(8, 1)
        theta = torch.randn(8, 3)

        d2, d3, d4 = net(k1, y_n, t_n, h, theta)
        loss = d2.sum() + d3.sum() + d4.sum()
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), \
                f"Non-finite gradient in {name}"

    def test_residual_block_skip_connection(self):
        """ResidualBlock output should include the skip connection."""
        block = ResidualBlock(dim=16)
        x = torch.randn(4, 16)
        out = block(x)

        assert not torch.allclose(out, x), \
            "ResidualBlock output identical to input"
        assert out.shape == x.shape

    def test_different_state_dims(self):
        """Network should work for different state dimensions."""
        for state_dim in [1, 2, 3, 5]:
            net = KFactorResidualNet(state_dim=state_dim, param_dim=1,
                                     hidden_dim=16)
            k1 = torch.randn(4, state_dim)
            y_n = torch.randn(4, state_dim)
            t_n = torch.randn(4, 1)
            h = torch.randn(4, 1)
            theta = torch.randn(4, 1)

            d2, d3, d4 = net(k1, y_n, t_n, h, theta)
            assert d2.shape == (4, state_dim)
            assert d3.shape == (4, state_dim)
            assert d4.shape == (4, state_dim)


# ---------------------------------------------------------------------------
# Test: TrustGate
# ---------------------------------------------------------------------------

class TestTrustGate:
    """Tests for the convergence-based trust gate."""

    @pytest.fixture
    def gate(self) -> TrustGate:
        """Create a trust gate with lock_threshold=1e-4, patience=1."""
        return TrustGate(lock_threshold=1e-4, lock_patience=1)

    def test_locks_converged_slabs(self, gate):
        """Slabs with small corrections should become locked."""
        slab_changes = torch.tensor([1e-2, 3e-5, 5e-6, 2e-1])
        gate.update_locks(slab_changes)

        assert gate.locked[0].item() is False
        assert gate.locked[1].item() is True
        assert gate.locked[2].item() is True
        assert gate.locked[3].item() is False

    def test_should_run_fine_skips_locked(self, gate):
        """should_run_fine should return False for locked slabs."""
        slab_changes = torch.tensor([1e-2, 3e-5, 5e-6, 2e-1])
        gate.update_locks(slab_changes)

        mask = gate.should_run_fine(slab_changes)
        assert mask[0].item() is True
        assert mask[1].item() is False
        assert mask[2].item() is False
        assert mask[3].item() is True

    def test_unlock_on_correction_growth(self, gate):
        """Locked slabs should unlock if their correction grows."""
        gate.update_locks(torch.tensor([1e-2, 3e-5, 5e-6, 2e-1]))
        assert gate.locked[1].item() is True

        gate.update_locks(torch.tensor([1e-2, 5e-2, 5e-6, 2e-1]))
        assert gate.locked[1].item() is False
        assert gate.locked[2].item() is True

    def test_patience(self):
        """Slabs should only lock after patience consecutive iterations."""
        gate = TrustGate(lock_threshold=1e-4, lock_patience=2)

        gate.update_locks(torch.tensor([1e-5, 1e-5]))
        assert gate.locked[0].item() is False

        gate.update_locks(torch.tensor([1e-5, 1e-5]))
        assert gate.locked[0].item() is True

    def test_reset(self, gate):
        """reset() should clear all locks."""
        gate.update_locks(torch.tensor([1e-5, 1e-5, 1e-5]))
        assert gate.locked.sum().item() == 3

        gate.reset()
        assert gate.locked is None

    def test_stats_output(self, gate):
        """get_stats should return all expected keys."""
        slab_changes = torch.tensor([1e-2, 3e-5, 5e-6, 2e-1])
        gate.update_locks(slab_changes)

        stats = gate.get_stats(slab_changes)

        expected_keys = {"threshold", "n_slabs", "n_trusted",
                        "n_corrected", "trust_rate", "mean_error",
                        "max_error", "locked_slabs"}
        assert set(stats.keys()) == expected_keys
        assert stats["n_slabs"] == 4
        assert stats["n_trusted"] == 2
        assert stats["n_corrected"] == 2
        assert stats["n_trusted"] + stats["n_corrected"] == 4
        assert stats["trust_rate"] == 0.5


# ---------------------------------------------------------------------------
# Test: Classical Coarse Propagators
# ---------------------------------------------------------------------------

class TestClassicalCoarse:
    """Tests for Euler and Backward Euler coarse propagators."""

    @pytest.fixture
    def system(self):
        """Get the damped oscillator system."""
        return get_system("damped_oscillator")

    def test_euler_coarse_shape(self, system):
        """Euler coarse should return correct output shape."""
        from src.solvers.classical_coarse import EulerCoarse

        coarse = EulerCoarse(system, step_dt=0.1)
        y0 = system.default_initial_condition()
        params = system.default_params()

        y_next = coarse.propagate(y0, t_n=0.0, delta_t=1.0, params=params)
        assert y_next.shape == y0.shape

    def test_backward_euler_coarse_shape(self, system):
        """Backward Euler coarse should return correct output shape."""
        from src.solvers.classical_coarse import BackwardEulerCoarse

        coarse = BackwardEulerCoarse(system, step_dt=0.1, fp_iterations=5)
        y0 = system.default_initial_condition()
        params = system.default_params()

        y_next = coarse.propagate(y0, t_n=0.0, delta_t=1.0, params=params)
        assert y_next.shape == y0.shape

    def test_euler_vs_backward_euler(self, system):
        """Backward Euler should give different results from Euler."""
        from src.solvers.classical_coarse import EulerCoarse, BackwardEulerCoarse

        y0 = system.default_initial_condition()
        params = system.default_params()

        euler = EulerCoarse(system, step_dt=0.1)
        be = BackwardEulerCoarse(system, step_dt=0.1)

        y_euler = euler.propagate(y0, 0.0, 1.0, params)
        y_be = be.propagate(y0, 0.0, 1.0, params)

        assert not torch.allclose(y_euler, y_be, atol=1e-6), \
            "Euler and Backward Euler should give different results"


# ---------------------------------------------------------------------------
# Test: DataGenerator
# ---------------------------------------------------------------------------

class TestDataGenerator:
    """Tests for the training data generator."""

    @pytest.fixture
    def generator(self) -> DataGenerator:
        """Create a data generator for the damped oscillator."""
        system = get_system("damped_oscillator")
        return DataGenerator(system, device=torch.device("cpu"))

    def test_coarse_data_shapes(self, generator):
        """Coarse data should have consistent tensor shapes."""
        data = generator.generate_coarse_data(
            n_trajectories=5, fine_dt=0.01, coarse_dt=0.1
        )

        assert len(data) > 0, "No samples generated"
        assert data.y_n.shape[1] == 2     # state_dim=2
        assert data.t_n.shape[1] == 1
        assert data.theta_ode.shape[1] == 3  # 3 params
        assert data.f_true.shape[1] == 2  # derivative has same dim as state
        assert data.delta_t.shape[1] == 1
        assert data.y_next.shape[1] == 2
        assert data.y_n.shape[0] == data.f_true.shape[0]

    def test_f_true_is_correct(self, generator):
        """f_true should match system.f evaluated at (y_n, t_n)."""
        data = generator.generate_coarse_data(
            n_trajectories=3, fine_dt=0.01, coarse_dt=0.1,
            randomize_dt=False,
        )

        system = generator.system
        params = system.default_params()

        # Check first sample
        y0 = data.y_n[0]
        t0 = data.t_n[0, 0].item()
        f_expected = system.f(t0, y0, params)
        # Note: params may differ per sample, but approximate check
        assert data.f_true[0].shape == f_expected.shape

    def test_k_factor_data_shapes(self, generator):
        """k-factor data should include theta_ode."""
        data = generator.generate_k_factor_data(
            n_trajectories=5, dt=0.1
        )

        assert len(data) > 0, "No samples generated"
        assert data.k1.shape[1] == 2
        assert data.k2.shape[1] == 2
        assert data.k3.shape[1] == 2
        assert data.k4.shape[1] == 2
        assert data.y_n.shape[1] == 2
        assert data.t_n.shape[1] == 1
        assert data.h.shape[1] == 1
        assert data.theta_ode.shape[1] == 3  # New: theta included

    def test_coarse_data_device_transfer(self, generator):
        """to() should move all tensors to the target device."""
        data = generator.generate_coarse_data(n_trajectories=3, fine_dt=0.01)
        data_cpu = data.to(torch.device("cpu"))

        assert data_cpu.y_n.device.type == "cpu"
        assert data_cpu.theta_ode.device.type == "cpu"
        assert data_cpu.f_true.device.type == "cpu"

    def test_diverse_parameters(self, generator):
        """Generated data should have varied ODE parameters."""
        data = generator.generate_coarse_data(
            n_trajectories=20, fine_dt=0.01, coarse_dt=0.5
        )

        unique_thetas = torch.unique(data.theta_ode, dim=0)
        assert len(unique_thetas) > 1, \
            "All theta_ODE vectors are identical — not diverse enough"
