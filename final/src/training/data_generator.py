"""Training data generation from classical RK4 trajectories.

Generates two types of training data by running high-accuracy classical
RK4 on diverse trajectories (varied initial conditions and ODE parameters):

1. **Coarse propagator data**: (y_n, t_n, theta_ODE) → f_true(y_n, t_n)
   Pairs for training the derivative-predicting meta-propagator.
   Also includes delta_t and y_next for physics residual computation.

2. **k-factor data**: (k1, y_n, t_n, h, theta_ODE) → (k2, k3, k4)
   Pairs for training the k-factor residual network to predict
   corrections to RK4 stages.

Data diversity:
    Both initial conditions AND ODE parameters are sampled from ranges
    defined by each ODE system's ``param_ranges()`` method.  This ensures
    the meta-propagator learns a general mapping across the parameter
    family, not just one fixed configuration.

    Additionally, coarse_dt is randomized within [0.05, 0.2] to make
    the network robust to step size variations during multi-step
    coarse propagation.

Example:
    >>> from src.ode_systems import get_system
    >>> from src.training.data_generator import DataGenerator
    >>> system = get_system("damped_oscillator")
    >>> gen = DataGenerator(system)
    >>> coarse_data = gen.generate_coarse_data(n_trajectories=100)
    >>> k_data = gen.generate_k_factor_data(n_trajectories=100)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from src.ode_systems import ODESystem
from src.solvers.classical_rk4 import ClassicalRK4Solver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CoarseTrainingData:
    """Training data for the derivative-predicting coarse propagator.

    All tensors have shape ``(n_samples, ...)``.

    Attributes:
        y_n: Current states, shape ``(n_samples, state_dim)``.
        t_n: Current times, shape ``(n_samples, 1)``.
        theta_ode: ODE parameter vectors, shape ``(n_samples, param_dim)``.
        f_true: Exact derivatives at (y_n, t_n), shape ``(n_samples, state_dim)``.
        delta_t: Coarse time steps, shape ``(n_samples, 1)``.
            Used for physics residual computation during training.
        y_next: Target next states, shape ``(n_samples, state_dim)``.
            Used for physics residual and validation.
    """
    y_n: Tensor
    t_n: Tensor
    theta_ode: Tensor
    f_true: Tensor
    delta_t: Tensor
    y_next: Tensor

    def __len__(self) -> int:
        return self.y_n.shape[0]

    def to(self, device: torch.device) -> "CoarseTrainingData":
        """Move all tensors to the specified device.

        Args:
            device: Target torch device.

        Returns:
            New ``CoarseTrainingData`` with tensors on the target device.
        """
        return CoarseTrainingData(
            y_n=self.y_n.to(device),
            t_n=self.t_n.to(device),
            theta_ode=self.theta_ode.to(device),
            f_true=self.f_true.to(device),
            delta_t=self.delta_t.to(device),
            y_next=self.y_next.to(device),
        )


@dataclass
class KFactorTrainingData:
    """Training data for the k-factor residual network.

    All tensors have shape ``(n_samples, ...)``.

    Attributes:
        k1: Exact first RK4 slopes, shape ``(n_samples, state_dim)``.
        y_n: Current states, shape ``(n_samples, state_dim)``.
        t_n: Current times, shape ``(n_samples, 1)``.
        h: Step sizes, shape ``(n_samples, 1)``.
        theta_ode: ODE parameter vectors, shape ``(n_samples, param_dim)``.
        k2: Target second slopes, shape ``(n_samples, state_dim)``.
        k3: Target third slopes, shape ``(n_samples, state_dim)``.
        k4: Target fourth slopes, shape ``(n_samples, state_dim)``.
    """
    k1: Tensor
    y_n: Tensor
    t_n: Tensor
    h: Tensor
    theta_ode: Tensor
    k2: Tensor
    k3: Tensor
    k4: Tensor

    def __len__(self) -> int:
        return self.k1.shape[0]

    def to(self, device: torch.device) -> "KFactorTrainingData":
        """Move all tensors to the specified device.

        Args:
            device: Target torch device.

        Returns:
            New ``KFactorTrainingData`` with tensors on the target device.
        """
        return KFactorTrainingData(
            k1=self.k1.to(device),
            y_n=self.y_n.to(device),
            t_n=self.t_n.to(device),
            h=self.h.to(device),
            theta_ode=self.theta_ode.to(device),
            k2=self.k2.to(device),
            k3=self.k3.to(device),
            k4=self.k4.to(device),
        )


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------

class DataGenerator:
    """Generates training data by running classical RK4 on diverse trajectories.

    The generator samples random initial conditions and ODE parameters
    from the ranges defined by the ODE system, runs high-accuracy RK4
    trajectories, and collects (input, target) pairs for both the coarse
    propagator and k-factor residual networks.

    Attributes:
        system: The ODE system to generate data for.
        solver: Classical RK4 solver instance.
        device: Torch device for computation.

    Example:
        >>> system = get_system("damped_oscillator")
        >>> gen = DataGenerator(system, device=torch.device("cpu"))
        >>> coarse_data = gen.generate_coarse_data(
        ...     n_trajectories=200, fine_dt=0.001, coarse_dt=0.1)
    """

    def __init__(
        self,
        system: ODESystem,
        device: torch.device | None = None,
    ):
        """Initialise the data generator.

        Args:
            system: ODE system instance defining the dynamics and
                    parameter ranges.
            device: Torch device.  Defaults to CPU.
        """
        self.system = system
        self.device = device or torch.device("cpu")
        self.solver = ClassicalRK4Solver(device=self.device)

        logger.info(
            "DataGenerator created for system='%s' on device=%s",
            system.name, self.device,
        )

    def _sample_params(self) -> Dict[str, float]:
        """Sample random ODE parameters from the system's ranges.

        Each parameter is drawn uniformly from its defined range.

        Returns:
            Dictionary mapping parameter names to sampled scalar values.
        """
        params = {}
        for name, (low, high) in self.system.param_ranges().items():
            params[name] = low + (high - low) * torch.rand(1).item()
        return params

    def _sample_initial_condition(
        self,
        scale: float = 0.5,
    ) -> Tensor:
        """Sample a random initial condition near the system's default.

        Perturbs the default IC with uniform noise scaled by ``scale``.

        Args:
            scale: Magnitude of the perturbation relative to the default.
                   A value of 0.5 means each component is perturbed by
                   up to ±50% of its default value.

        Returns:
            Perturbed IC tensor of shape ``(state_dim,)``.
        """
        default_ic = self.system.default_initial_condition(device=self.device)
        perturbation = (2.0 * torch.rand_like(default_ic) - 1.0) * scale
        # Ensure non-zero by adding a small offset
        perturbed = default_ic * (1.0 + perturbation)
        return perturbed

    def generate_coarse_data(
        self,
        n_trajectories: int = 200,
        fine_dt: float = 0.001,
        coarse_dt: float = 0.1,
        ic_scale: float = 0.5,
        randomize_dt: bool = True,
        dt_range: Tuple[float, float] = (0.05, 0.2),
    ) -> CoarseTrainingData:
        """Generate training data for the derivative-predicting coarse propagator.

        For each trajectory:
            1. Sample random IC and ODE parameters.
            2. Run high-accuracy RK4 with step size ``fine_dt``.
            3. Sub-sample the trajectory at coarse intervals.
            4. At each sample point, compute the exact derivative
               f_true = system.f(t_n, y_n, params).
            5. Store (y_n, t_n, theta_ODE, f_true, delta_t, y_next) tuples.

        Args:
            n_trajectories: Number of diverse trajectories to generate.
            fine_dt: Small step size for the ground-truth RK4 solver.
            coarse_dt: Base coarse time step.  If ``randomize_dt`` is True,
                      this is the median of the random range.
            ic_scale: Scale for initial condition perturbation.
            randomize_dt: If True, randomize the coarse step size per
                         trajectory to make the model dt-robust.
            dt_range: (min_dt, max_dt) for randomized coarse steps.

        Returns:
            ``CoarseTrainingData`` containing all collected samples.
        """
        t_start, t_end = self.system.default_time_span()

        all_y_n = []
        all_t_n = []
        all_theta = []
        all_f_true = []
        all_delta_t = []
        all_y_next = []

        logger.info(
            "Generating coarse training data: n_traj=%d, fine_dt=%.4f, "
            "coarse_dt=%.3f%s, t=[%.1f, %.1f]",
            n_trajectories, fine_dt, coarse_dt,
            f" (randomized [{dt_range[0]:.2f}, {dt_range[1]:.2f}])"
            if randomize_dt else "",
            t_start, t_end,
        )

        for traj_idx in tqdm(range(n_trajectories), desc="Coarse data"):
            # Sample random params and IC
            params = self._sample_params()
            y0 = self._sample_initial_condition(scale=ic_scale)
            theta = self.system.param_vector(params, device=self.device)

            # Choose dt for this trajectory
            if randomize_dt:
                traj_dt = (
                    dt_range[0]
                    + (dt_range[1] - dt_range[0]) * torch.rand(1).item()
                )
            else:
                traj_dt = coarse_dt

            # Run fine RK4
            result = self.solver.solve_single(
                f=self.system.f, y0=y0,
                t_span=(t_start, t_end),
                dt=fine_dt, params=params,
            )

            # Sub-sample at coarse intervals
            coarse_step_ratio = max(1, int(round(traj_dt / fine_dt)))
            n_fine_steps = result.y.shape[0] - 1
            actual_dt = coarse_step_ratio * fine_dt

            for i in range(0, n_fine_steps - coarse_step_ratio, coarse_step_ratio):
                y_curr = result.y[i]
                y_next = result.y[i + coarse_step_ratio]
                t_curr = result.t[i].item()

                # Compute exact derivative at this point
                f_exact = self.system.f(t_curr, y_curr, params)

                all_y_n.append(y_curr)
                all_t_n.append(torch.tensor([t_curr], device=self.device))
                all_theta.append(theta)
                all_f_true.append(f_exact)
                all_delta_t.append(
                    torch.tensor([actual_dt], device=self.device)
                )
                all_y_next.append(y_next)

        data = CoarseTrainingData(
            y_n=torch.stack(all_y_n),
            t_n=torch.stack(all_t_n),
            theta_ode=torch.stack(all_theta),
            f_true=torch.stack(all_f_true),
            delta_t=torch.stack(all_delta_t),
            y_next=torch.stack(all_y_next),
        )

        logger.info(
            "Coarse data generated: %d samples from %d trajectories",
            len(data), n_trajectories,
        )
        return data

    def generate_k_factor_data(
        self,
        n_trajectories: int = 200,
        dt: float = 0.01,
        ic_scale: float = 0.5,
    ) -> KFactorTrainingData:
        """Generate training data for the k-factor residual network.

        For each trajectory:
            1. Sample random IC and ODE parameters.
            2. Run RK4 with ``return_k_factors=True``.
            3. Store (k1, y_n, t_n, h, theta_ode) → (k2, k3, k4) pairs.

        Args:
            n_trajectories: Number of diverse trajectories to generate.
            dt: Step size for RK4 (also becomes the ``h`` in training data).
            ic_scale: Scale for initial condition perturbation.

        Returns:
            ``KFactorTrainingData`` containing all collected samples.
        """
        t_start, t_end = self.system.default_time_span()

        all_k1, all_y_n, all_t_n, all_h, all_theta = [], [], [], [], []
        all_k2, all_k3, all_k4 = [], [], []

        logger.info(
            "Generating k-factor training data: n_traj=%d, dt=%.4f, "
            "t=[%.1f, %.1f]",
            n_trajectories, dt, t_start, t_end,
        )

        h_tensor = torch.tensor([dt], device=self.device)

        for traj_idx in tqdm(range(n_trajectories), desc="K-factor data"):
            params = self._sample_params()
            y0 = self._sample_initial_condition(scale=ic_scale)
            theta = self.system.param_vector(params, device=self.device)

            # Run RK4 with k-factor recording
            result = self.solver.solve_single(
                f=self.system.f, y0=y0,
                t_span=(t_start, t_end),
                dt=dt, params=params,
                return_k_factors=True,
            )

            # Collect k-factor data from each step
            for step_idx, (k1, k2, k3, k4) in enumerate(result.k_factors):
                y_curr = result.y[step_idx]
                t_curr = result.t[step_idx].item()

                all_k1.append(k1)
                all_y_n.append(y_curr)
                all_t_n.append(torch.tensor([t_curr], device=self.device))
                all_h.append(h_tensor)
                all_theta.append(theta)
                all_k2.append(k2)
                all_k3.append(k3)
                all_k4.append(k4)

        data = KFactorTrainingData(
            k1=torch.stack(all_k1),
            y_n=torch.stack(all_y_n),
            t_n=torch.stack(all_t_n),
            h=torch.stack(all_h),
            theta_ode=torch.stack(all_theta),
            k2=torch.stack(all_k2),
            k3=torch.stack(all_k3),
            k4=torch.stack(all_k4),
        )

        logger.info(
            "K-factor data generated: %d samples from %d trajectories",
            len(data), n_trajectories,
        )
        return data
