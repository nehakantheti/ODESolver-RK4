"""Solver modules for the Neural-Accelerated Parallel RK4 system.

Provides classical and neural-augmented ODE solvers:
    classical_rk4: Pure PyTorch RK4 with CPU, GPU, and vmap-batched modes.
    neural_augmented_rk4: RK4 enhanced with k-factor residual prediction.
    parareal: Parallel-in-time solver with neural coarse propagator.
    gpu_engine: GPU optimization utilities (AMP, torch.compile, batching).
"""
