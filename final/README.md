# Neural-Accelerated Parallel RK4 ODE Solver

A GPU-native, neural network-augmented, parallel-in-time ODE solver combining **Parareal decomposition**, **neural coarse propagation**, **k-factor residual prediction**, and **adaptive trust gating**.

## Key Innovation

| Component | What it does | Why it matters |
|-----------|-------------|----------------|
| **Meta-Propagator** | Neural coarse solver conditioned on θ_ODE | Generalises across ODE parameter families — no per-problem retraining |
| **k-Factor Residual Net** | Predicts corrections δ₂, δ₃, δ₄ from exact k₁ | Replaces 3 sequential f() evaluations with 1 NN forward pass |
| **Adaptive Trust Gate** | Per-slab confidence gating with geometric decay | Selectively skips fine corrections → fewer fine solves → faster convergence |
| **GPU-Native Parareal** | `torch.vmap` batched fine pass | Entire pipeline runs on GPU without CPU-GPU transfer bottlenecks |

## Architecture

```
Input: y₀, [t₀, T], ODE params θ
    │
    ├─── PARTITION: Split time into P slabs ──────────────────────┐
    │                                                              │
    ├─── COARSE PASS (sequential, fast):                          │
    │    Meta-Propagator NN([y_n, t, Δt, θ]) → ŷ_{n+1}, ε_n     │
    │                                                              │
    ├─── ITERATE until convergence:                               │
    │    │                                                         │
    │    ├── TRUST GATE: ε_n < τ → skip fine, else:              │
    │    │                                                         │
    │    ├── FINE PASS (parallel, GPU-batched):                   │
    │    │   Classical RK4 on each active slab                     │
    │    │   (optionally with k-factor acceleration)              │
    │    │                                                         │
    │    └── CORRECTION: U^{k+1} = G_new + (F_old - G_old)       │
    │                                                              │
    └─── OUTPUT: y(t) at all slab boundaries ─────────────────────┘
```

## Benchmark ODE Systems

| System | State dim | Type | Key Challenge |
|--------|-----------|------|---------------|
| Damped Harmonic Oscillator | 2 | Linear | Baseline with analytical solution |
| Lotka-Volterra | 2 | Nonlinear | Oscillatory predator-prey dynamics |
| Van der Pol | 2 | Nonlinear | Limit cycles, mildly stiff |
| Lorenz Attractor | 3 | Chaotic | Sensitive dependence on ICs |

## Quick Start

### Requirements

- Python 3.11
- PyTorch ≥ 2.1.0 (CUDA optional but recommended)

### Installation

```bash
cd final
py -3.11 -m pip install -r requirements.txt
```

### Run Tests

```bash
py -3.11 -m pytest tests/ -v
```

### Train Neural Components

```bash
# Train for a specific ODE system
py -3.11 -m src.training.train_all --system damped_oscillator

# Options: damped_oscillator, lotka_volterra, van_der_pol, lorenz
```

### Launch Demo Dashboard

```bash
py -3.11 -m streamlit run demo/app.py
```

### Run Benchmarks

```bash
py -3.11 benchmarks/benchmark_solvers.py
```

## Project Structure

```
final/
├── requirements.txt            # Dependencies (Python 3.11)
├── DEVLOG.md                   # Development log
├── src/
│   ├── ode_systems.py          # 4 benchmark ODE systems (ABC pattern)
│   ├── solvers/
│   │   ├── classical_rk4.py    # Pure PyTorch RK4 (single, batched, interval)
│   │   └── parareal.py         # Parareal algorithm with trust gate
│   ├── networks/
│   │   ├── coarse_propagator.py # Meta-propagator NN (θ_ODE conditioned)
│   │   ├── k_factor_residual.py # Residual correction network
│   │   └── trust_gate.py        # Adaptive gating mechanism
│   ├── training/
│   │   ├── data_generator.py    # Diverse trajectory data generation
│   │   ├── train_coarse.py      # Coarse propagator training pipeline
│   │   ├── train_k_factor.py    # k-factor network training pipeline
│   │   └── train_all.py         # End-to-end training orchestrator
│   └── visualization/
│       └── plots.py             # Dark-theme plotting utilities
├── demo/
│   └── app.py                   # 4-tab Streamlit dashboard
├── benchmarks/
│   └── benchmark_solvers.py     # Performance measurement suite
├── tests/
│   ├── test_rk4_correctness.py  # 21 tests (shapes, analytical, scipy)
│   ├── test_networks.py         # 19 tests (NN shapes, gradients, gate)
│   └── test_parareal_convergence.py # 6 tests (convergence, accuracy)
└── trained_models/              # Saved model weights
```

## SOLID Principles Applied

- **SRP**: Each module has one responsibility (e.g., `classical_rk4.py` only solves, `data_generator.py` only generates data)
- **OCP**: New ODE systems via `ODESystem` subclassing; new solvers via the same interface
- **LSP**: All `ODESystem` subclasses are substitutable through the abstract interface
- **ISP**: Solver methods are granular (`solve_single`, `solve_batched`, `solve_interval`)
- **DIP**: Solvers depend on `ODESystem` abstraction and `DerivativeFunc` type alias, not concrete classes

## References

See [RESEARCH_REFERENCES.md](../RESEARCH_REFERENCES.md) for the full list of 24 papers and resources.

## License

Academic project — see repository root for license information.
