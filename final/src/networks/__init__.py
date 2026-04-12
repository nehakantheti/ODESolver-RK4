"""Neural network architectures for the solver pipeline.

Provides:
    coarse_propagator: Meta-propagator NN conditioned on ODE parameters.
    k_factor_residual: Residual network predicting k-factor corrections.
    trust_gate: Confidence estimation and adaptive gating logic.
"""
