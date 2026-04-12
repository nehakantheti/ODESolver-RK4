"""Benchmark ODE system definitions for solver evaluation.

Defines four ODE systems of increasing complexity, each as a class
implementing the ``ODESystem`` abstract interface. Every system exposes
its derivative function ``f(t, y, params)``, default parameters, initial
conditions, and (where available) an analytical solution for validation.

Systems:
    1. DampedHarmonicOscillator — linear, analytical solution exists.
    2. LotkaVolterra — nonlinear predator-prey, oscillatory.
    3. VanDerPolOscillator — limit cycles, mildly stiff for large mu.
    4. LorenzAttractor — chaotic, sensitive to initial conditions.

Design:
    All systems accept an explicit ``params`` dict so the same class can
    represent an entire *family* of ODEs.  This is critical for the
    meta-propagator, which is conditioned on ``theta_ode`` (the parameter
    vector) to generalise across problem instances.

Example:
    >>> system = DampedHarmonicOscillator()
    >>> y0 = system.default_initial_condition()
    >>> params = system.default_params()
    >>> dydt = system.f(t=0.0, y=y0, params=params)
"""

from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ODESystem(abc.ABC):
    """Abstract base class for an ODE system  dy/dt = f(t, y; params).

    Sub-classes must implement ``f``, ``default_params``,
    ``default_initial_condition``, and ``default_time_span``.

    Attributes:
        name: Human-readable name of the ODE system.
        dim: Dimensionality of the state vector y.
        param_names: Ordered list of parameter names that ``params`` expects.
    """

    name: str
    dim: int
    param_names: List[str]

    # -- abstract interface --------------------------------------------------

    @abc.abstractmethod
    def f(self, t: float, y: Tensor, params: Dict[str, float]) -> Tensor:
        """Compute the derivative dy/dt at a given state.

        Args:
            t: Current time.
            y: State vector of shape ``(dim,)`` or ``(batch, dim)``.
            params: Dictionary mapping ``param_names`` to scalar values.

        Returns:
            Derivative tensor with the same shape as ``y``.
        """

    @abc.abstractmethod
    def default_params(self) -> Dict[str, float]:
        """Return the default parameter dictionary for this system.

        Returns:
            Dictionary mapping each name in ``param_names`` to its default
            scalar value.
        """

    @abc.abstractmethod
    def default_initial_condition(self, device: torch.device | None = None) -> Tensor:
        """Return the default initial condition y(t0).

        Args:
            device: Torch device for the returned tensor.

        Returns:
            1-D tensor of shape ``(dim,)``.
        """

    @abc.abstractmethod
    def default_time_span(self) -> Tuple[float, float]:
        """Return the default integration interval ``(t_start, t_end)``.

        Returns:
            Tuple of two floats.
        """

    # -- optional ------------------------------------------------------------

    def analytical_solution(
        self, t: Tensor, params: Dict[str, float]
    ) -> Optional[Tensor]:
        """Return the exact analytical solution at times ``t``, if known.

        Args:
            t: 1-D tensor of time points.
            params: Parameter dictionary.

        Returns:
            Tensor of shape ``(len(t), dim)`` or ``None`` if no closed-form
            solution is available.
        """
        return None

    def param_vector(self, params: Dict[str, float], device: torch.device | None = None) -> Tensor:
        """Convert the parameter dict to an ordered 1-D tensor (theta_ode).

        The ordering follows ``self.param_names``.  This tensor is fed to
        the meta-propagator network as conditioning input.

        Args:
            params: Parameter dictionary.
            device: Torch device for the returned tensor.

        Returns:
            1-D tensor of shape ``(len(param_names),)``.
        """
        values = [params[name] for name in self.param_names]
        return torch.tensor(values, dtype=torch.float32, device=device)

    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return training-data sampling ranges for each parameter.

        Returns:
            Dictionary mapping each parameter name to a ``(low, high)``
            tuple used by the data generator to create diverse trajectories.
        """
        return {name: (val * 0.5, val * 2.0)
                for name, val in self.default_params().items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', dim={self.dim})"


# ---------------------------------------------------------------------------
# System 1 — Damped Harmonic Oscillator
# ---------------------------------------------------------------------------

class DampedHarmonicOscillator(ODESystem):
    """Damped harmonic oscillator: m·x'' + c·x' + k·x = 0.

    Rewritten as a first-order system with state ``y = [x, v]``:
        dx/dt = v
        dv/dt = -(k/m)·x - (c/m)·v

    This system is linear, has an analytical solution for the under-damped
    case (c² < 4mk), and provides continuity with the ``mid/phase1/``
    baseline code.

    Attributes:
        name: ``"Damped Harmonic Oscillator"``
        dim: 2  (position x, velocity v)
        param_names: ``["mass", "damping", "stiffness"]``
    """

    name = "Damped Harmonic Oscillator"
    dim = 2
    param_names = ["mass", "damping", "stiffness"]

    def f(self, t: float, y: Tensor, params: Dict[str, float]) -> Tensor:
        """Compute [dx/dt, dv/dt] for the damped oscillator.

        Args:
            t: Current time (unused for this autonomous system, kept for
               interface consistency).
            y: State ``[x, v]`` of shape ``(2,)`` or ``(batch, 2)``.
            params: Must contain ``mass``, ``damping``, ``stiffness``.

        Returns:
            Derivative tensor of the same shape as ``y``.
        """
        m = params["mass"]
        c = params["damping"]
        k = params["stiffness"]

        # Handle both single (2,) and batched (batch, 2) inputs
        x = y[..., 0]
        v = y[..., 1]

        dxdt = v
        dvdt = -(k / m) * x - (c / m) * v

        return torch.stack([dxdt, dvdt], dim=-1)

    def default_params(self) -> Dict[str, float]:
        """Default: m=1, c=0.1, k=1 (under-damped oscillation)."""
        return {"mass": 1.0, "damping": 0.1, "stiffness": 1.0}

    def default_initial_condition(self, device: torch.device | None = None) -> Tensor:
        """Default IC: x(0)=1, v(0)=0 (released from displacement)."""
        return torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)

    def default_time_span(self) -> Tuple[float, float]:
        """Default: integrate from t=0 to t=20."""
        return (0.0, 20.0)

    def analytical_solution(
        self, t: Tensor, params: Dict[str, float]
    ) -> Optional[Tensor]:
        """Exact solution for the under-damped case (c² < 4mk).

        The general solution is:
            x(t) = A·exp(-γt)·cos(ωd·t − φ)

        where γ = c/(2m), ωd = √(k/m − γ²), and A, φ are determined
        by initial conditions x(0)=1, v(0)=0.

        Args:
            t: 1-D tensor of time points.
            params: Must contain ``mass``, ``damping``, ``stiffness``.

        Returns:
            Tensor of shape ``(len(t), 2)`` containing ``[x(t), v(t)]``,
            or ``None`` if the system is not under-damped.
        """
        m = params["mass"]
        c = params["damping"]
        k = params["stiffness"]

        gamma = c / (2.0 * m)
        omega_0_sq = k / m

        discriminant = omega_0_sq - gamma ** 2
        if discriminant <= 0:
            logger.warning(
                "Analytical solution only available for under-damped case "
                "(c² < 4mk). Got discriminant=%.4f", discriminant
            )
            return None

        omega_d = math.sqrt(discriminant)

        # IC: x(0) = 1, v(0) = 0
        # x(t) = exp(-γt) * [cos(ωd·t) + (γ/ωd)·sin(ωd·t)]
        # v(t) = dx/dt
        exp_term = torch.exp(-gamma * t)
        cos_term = torch.cos(omega_d * t)
        sin_term = torch.sin(omega_d * t)

        x = exp_term * (cos_term + (gamma / omega_d) * sin_term)
        v = exp_term * (
            -(gamma + omega_d * (gamma / omega_d)) * cos_term
            + (omega_d - gamma * (gamma / omega_d)) * sin_term
        )
        # Simplify v:
        # v = exp(-γt) * [(-ω₀²/ωd)·sin(ωd·t)]
        v = exp_term * (-(omega_0_sq / omega_d) * sin_term)

        return torch.stack([x, v], dim=-1)

    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Ranges for diverse data generation (under-damped regime)."""
        return {
            "mass": (0.5, 2.0),
            "damping": (0.01, 0.5),
            "stiffness": (0.5, 2.0),
        }


# ---------------------------------------------------------------------------
# System 2 — Lotka-Volterra (Predator-Prey)
# ---------------------------------------------------------------------------

class LotkaVolterra(ODESystem):
    """Lotka-Volterra predator-prey model.

    Equations:
        dx/dt =  α·x  − β·x·y    (prey growth − predation)
        dy/dt =  δ·x·y − γ·y      (predator growth − death)

    This system is nonlinear, oscillatory, and conserves a Hamiltonian-like
    quantity (useful for verifying solver accuracy over long integrations).

    Attributes:
        name: ``"Lotka-Volterra"``
        dim: 2  (prey x, predator y)
        param_names: ``["alpha", "beta", "delta", "gamma"]``
    """

    name = "Lotka-Volterra"
    dim = 2
    param_names = ["alpha", "beta", "delta", "gamma"]

    def f(self, t: float, y: Tensor, params: Dict[str, float]) -> Tensor:
        """Compute [dx/dt, dy/dt] for the predator-prey system.

        Args:
            t: Current time (unused — autonomous system).
            y: State ``[prey, predator]`` of shape ``(2,)`` or ``(batch, 2)``.
            params: Must contain ``alpha``, ``beta``, ``delta``, ``gamma``.

        Returns:
            Derivative tensor of the same shape as ``y``.
        """
        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]
        gamma = params["gamma"]

        prey = y[..., 0]
        predator = y[..., 1]

        dprey_dt = alpha * prey - beta * prey * predator
        dpredator_dt = delta * prey * predator - gamma * predator

        return torch.stack([dprey_dt, dpredator_dt], dim=-1)

    def default_params(self) -> Dict[str, float]:
        """Default: classic Lotka-Volterra parameters."""
        return {"alpha": 1.5, "beta": 1.0, "delta": 1.0, "gamma": 3.0}

    def default_initial_condition(self, device: torch.device | None = None) -> Tensor:
        """Default IC: 10 prey, 5 predators."""
        return torch.tensor([10.0, 5.0], dtype=torch.float32, device=device)

    def default_time_span(self) -> Tuple[float, float]:
        """Default: t=0 to t=15 (several oscillation periods)."""
        return (0.0, 15.0)

    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Ranges ensuring oscillatory dynamics."""
        return {
            "alpha": (0.5, 2.5),
            "beta": (0.5, 1.5),
            "delta": (0.5, 1.5),
            "gamma": (1.0, 4.0),
        }


# ---------------------------------------------------------------------------
# System 3 — Van der Pol Oscillator
# ---------------------------------------------------------------------------

class VanDerPolOscillator(ODESystem):
    """Van der Pol oscillator: x'' − μ(1 − x²)x' + x = 0.

    Rewritten as a first-order system with state ``y = [x, v]``:
        dx/dt = v
        dv/dt = μ(1 − x²)v − x

    This system exhibits limit-cycle behaviour and becomes mildly stiff
    for large values of μ.  Our solver targets μ ∈ [0.1, 5.0] (non-stiff
    to mildly stiff regime where explicit RK4 is still viable).

    Attributes:
        name: ``"Van der Pol Oscillator"``
        dim: 2  (position x, velocity v)
        param_names: ``["mu"]``
    """

    name = "Van der Pol Oscillator"
    dim = 2
    param_names = ["mu"]

    def f(self, t: float, y: Tensor, params: Dict[str, float]) -> Tensor:
        """Compute [dx/dt, dv/dt] for the Van der Pol system.

        Args:
            t: Current time (unused — autonomous system).
            y: State ``[x, v]`` of shape ``(2,)`` or ``(batch, 2)``.
            params: Must contain ``mu`` (nonlinearity parameter).

        Returns:
            Derivative tensor of the same shape as ``y``.
        """
        mu = params["mu"]

        x = y[..., 0]
        v = y[..., 1]

        dxdt = v
        dvdt = mu * (1.0 - x ** 2) * v - x

        return torch.stack([dxdt, dvdt], dim=-1)

    def default_params(self) -> Dict[str, float]:
        """Default: mu=1.0 (moderate nonlinearity, non-stiff)."""
        return {"mu": 1.0}

    def default_initial_condition(self, device: torch.device | None = None) -> Tensor:
        """Default IC: x(0)=2, v(0)=0."""
        return torch.tensor([2.0, 0.0], dtype=torch.float32, device=device)

    def default_time_span(self) -> Tuple[float, float]:
        """Default: t=0 to t=20 (several limit-cycle periods)."""
        return (0.0, 20.0)

    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Ranges within the explicit-RK4-viable regime."""
        return {"mu": (0.1, 5.0)}


# ---------------------------------------------------------------------------
# System 4 — Lorenz Attractor
# ---------------------------------------------------------------------------

class LorenzAttractor(ODESystem):
    """Lorenz attractor (chaotic system).

    Equations:
        dx/dt = σ(y − x)
        dy/dt = x(ρ − z) − y
        dz/dt = xy − βz

    This 3-D system exhibits deterministic chaos for the classic parameter
    values σ=10, ρ=28, β=8/3.  It is a stringent stress test for any
    neural-augmented solver because small errors grow exponentially.

    Note:
        The Lorenz system is chaotic but **not** stiff.  Chaos means
        sensitivity to initial conditions; stiffness means disparate
        timescales.  Explicit RK4 handles Lorenz well with h ≤ 0.01.

    Attributes:
        name: ``"Lorenz Attractor"``
        dim: 3  (x, y, z)
        param_names: ``["sigma", "rho", "beta"]``
    """

    name = "Lorenz Attractor"
    dim = 3
    param_names = ["sigma", "rho", "beta"]

    def f(self, t: float, y: Tensor, params: Dict[str, float]) -> Tensor:
        """Compute [dx/dt, dy/dt, dz/dt] for the Lorenz system.

        Args:
            t: Current time (unused — autonomous system).
            y: State ``[x, y, z]`` of shape ``(3,)`` or ``(batch, 3)``.
            params: Must contain ``sigma``, ``rho``, ``beta``.

        Returns:
            Derivative tensor of the same shape as ``y``.
        """
        sigma = params["sigma"]
        rho = params["rho"]
        beta = params["beta"]

        x = y[..., 0]
        y_ = y[..., 1]  # avoid shadowing the parameter name
        z = y[..., 2]

        dxdt = sigma * (y_ - x)
        dydt = x * (rho - z) - y_
        dzdt = x * y_ - beta * z

        return torch.stack([dxdt, dydt, dzdt], dim=-1)

    def default_params(self) -> Dict[str, float]:
        """Default: classic chaotic parameters σ=10, ρ=28, β=8/3."""
        return {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}

    def default_initial_condition(self, device: torch.device | None = None) -> Tensor:
        """Default IC: [1, 1, 1] (near but not at the origin)."""
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    def default_time_span(self) -> Tuple[float, float]:
        """Default: t=0 to t=25 (long enough to see chaotic mixing)."""
        return (0.0, 25.0)

    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Ranges keeping the system in the chaotic regime."""
        return {
            "sigma": (8.0, 12.0),
            "rho": (25.0, 32.0),
            "beta": (2.0, 3.5),
        }


# ---------------------------------------------------------------------------
# Registry — convenience access to all systems
# ---------------------------------------------------------------------------

SYSTEM_REGISTRY: Dict[str, type] = {
    "damped_oscillator": DampedHarmonicOscillator,
    "lotka_volterra": LotkaVolterra,
    "van_der_pol": VanDerPolOscillator,
    "lorenz": LorenzAttractor,
}


def get_system(name: str) -> ODESystem:
    """Instantiate an ODE system by its registry key.

    Args:
        name: One of ``"damped_oscillator"``, ``"lotka_volterra"``,
              ``"van_der_pol"``, ``"lorenz"``.

    Returns:
        An instance of the corresponding ``ODESystem`` subclass.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    if name not in SYSTEM_REGISTRY:
        raise KeyError(
            f"Unknown ODE system '{name}'. "
            f"Available: {list(SYSTEM_REGISTRY.keys())}"
        )
    logger.info("Instantiating ODE system: %s", name)
    return SYSTEM_REGISTRY[name]()
