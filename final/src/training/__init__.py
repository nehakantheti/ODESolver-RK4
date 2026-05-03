"""Training pipelines for neural solver components.

Provides:
    data_generator: Generate training data from classical RK4 runs.
    train_coarse: Train the meta-propagator coarse solver.
    train_k_factor: Train the k-factor residual prediction network.
    train_all: End-to-end training orchestrator.
"""
