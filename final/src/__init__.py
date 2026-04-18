"""Neural-Accelerated Parallel RK4 ODE Solver.

A GPU-native, neural network-augmented, parallel-in-time RK4 ODE solver
combining Parareal decomposition, neural coarse propagation, k-factor
residual prediction, and adaptive trust gating.

Modules:
    ode_systems: Benchmark ODE system definitions.
    solvers: Classical and neural-augmented RK4 solvers.
    networks: Neural network architectures for coarse propagation,
              k-factor prediction, and trust gating.
    training: Data generation and training pipelines.
    visualization: Plotting and animation utilities.
"""

__version__ = "0.1.0"
