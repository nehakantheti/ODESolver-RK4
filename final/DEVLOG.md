# Neural-Accelerated Parallel RK4 ODE Solver — Development Log

> Single documentation file tracking all implementation progress.
> Updated incrementally as each component is built and tested.

---

## Project Structure

```
final/
├── README.md
├── requirements.txt
├── DEVLOG.md                     ← this file
├── src/
│   ├── __init__.py
│   ├── ode_systems.py            ✅ Implemented
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── classical_rk4.py      ✅ Implemented
│   │   ├── neural_augmented_rk4.py  ⬜ Pending
│   │   ├── parareal.py              ⬜ Pending
│   │   └── gpu_engine.py            ⬜ Pending
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── coarse_propagator.py     ⬜ Pending
│   │   ├── k_factor_residual.py     ⬜ Pending
│   │   └── trust_gate.py            ⬜ Pending
│   ├── training/
│   │   ├── __init__.py
│   │   ├── data_generator.py        ⬜ Pending
│   │   ├── train_coarse.py          ⬜ Pending
│   │   ├── train_k_factor.py        ⬜ Pending
│   │   └── train_all.py             ⬜ Pending
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                 ⬜ Pending
├── benchmarks/                      ⬜ Pending
├── demo/                            ⬜ Pending
├── tests/
│   ├── test_rk4_correctness.py   ✅ Implemented
│   ├── test_networks.py             ⬜ Pending
│   ├── test_parareal_convergence.py ⬜ Pending
│   └── test_trust_gate.py           ⬜ Pending
├── cpp_baseline/                    ⬜ Pending
└── trained_models/
```

---

## Phase 1 — Foundation

### Branch: `final/phase1-foundation`

### 1.1 ODE Systems (`src/ode_systems.py`)

**Status**: ✅ Complete

Implemented 4 benchmark ODE systems as subclasses of `ODESystem` ABC:

| System | State dim | Parameters | Key Feature |
|--------|-----------|------------|-------------|
| `DampedHarmonicOscillator` | 2 (x, v) | mass, damping, stiffness | Analytical solution available |
| `LotkaVolterra` | 2 (prey, pred) | α, β, δ, γ | Nonlinear, oscillatory |
| `VanDerPolOscillator` | 2 (x, v) | μ | Limit cycles; mildly stiff for large μ |
| `LorenzAttractor` | 3 (x, y, z) | σ, ρ, β | Chaotic stress test |

**Design principles applied**:
- **OCP (Open/Closed)**: New ODE systems can be added by subclassing `ODESystem` without modifying existing code.
- **LSP (Liskov Substitution)**: All systems are interchangeable through the `ODESystem` interface.
- **DIP (Dependency Inversion)**: Solvers depend on the `ODESystem` abstraction, not concrete implementations.

**Key features**:
- `param_vector()` method serialises params dict to a 1-D tensor (θ_ODE) for the meta-propagator network input.
- `param_ranges()` provides sampling bounds for diverse training data generation.
- `SYSTEM_REGISTRY` + `get_system()` for convenient access by string key.
- All derivative functions handle both single `(dim,)` and batched `(batch, dim)` inputs.

### 1.2 Classical RK4 Solver (`src/solvers/classical_rk4.py`)

**Status**: ✅ Complete

Implemented `ClassicalRK4Solver` with three execution modes:

| Method | Description | Use Case |
|--------|-------------|----------|
| `solve_single` | Sequential loop, full trajectory stored | Ground truth, validation |
| `solve_batched` | `torch.vmap` over batch of ICs | Parareal fine pass |
| `solve_interval` | Only returns endpoint (no storage) | Parareal fine endpoint only |

**Key features**:
- `rk4_step` static method exposes all 4 k-factors (k1, k2, k3, k4) — these become training data for the k-factor residual network.
- `RK4StepResult` and `SolveResult` dataclasses for clean return types.
- `solve_batched` attempts `torch.vmap` first, falls back to sequential loop if unsupported.
- Input validation with clear error messages.
- Logging at INFO (high-level) and DEBUG (per-step) levels.

### 1.3 Tests (`tests/test_rk4_correctness.py`)

**Status**: ✅ Complete

Test coverage:
- `TestODESystems`: Registry, dimensions, shapes, param_vector, param_ranges
- `TestClassicalRK4Basic`: Output shapes, IC preservation, k-factors, error handling
- `TestRK4VsAnalytical`: Position and velocity accuracy vs analytical solution (damped oscillator)
- `TestRK4VsScipy`: Cross-validation against scipy.integrate.solve_ivp for all 4 systems
- `TestBatchedSolve`: Batched results match single-solve results
- `TestSolveInterval`: Interval endpoint matches full solve endpoint

---

## Phase 2 — Neural Components

### Branch: `final/phase2-neural-components` (planned)

⬜ `src/networks/coarse_propagator.py` — Meta-propagator NN conditioned on θ_ODE
⬜ `src/networks/k_factor_residual.py` — Residual net: k₁ → δ₂, δ₃, δ₄
⬜ `src/networks/trust_gate.py` — Confidence head + gating logic
⬜ `src/training/data_generator.py` — Generate training data from classical RK4
⬜ `src/training/train_coarse.py` — Train meta-propagator
⬜ `src/training/train_k_factor.py` — Train k-factor residual net
⬜ `tests/test_networks.py` — Shape, range, gradient flow tests

---

## Phase 3 — Integration

### Branch: `final/phase3-integration` (planned)

⬜ `src/solvers/neural_augmented_rk4.py` — RK4 with k-factor gating
⬜ `src/solvers/parareal.py` — Full Parareal algorithm
⬜ `src/solvers/gpu_engine.py` — AMP, compile, vmap wrappers
⬜ `src/training/train_all.py` — End-to-end training orchestrator
⬜ `tests/test_parareal_convergence.py` — Convergence tests

---

## Phase 4 — Demo & Polish

### Branch: `final/phase4-demo` (planned)

⬜ `demo/app.py` — Streamlit dashboard
⬜ `benchmarks/benchmark_solvers.py` — Full speedup analysis
⬜ `benchmarks/benchmark_gpu.py` — GPU utilisation profiling
⬜ `src/visualization/plots.py` — Plotting utilities
⬜ Final `README.md`
