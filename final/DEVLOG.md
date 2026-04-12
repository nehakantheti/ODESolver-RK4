# Neural-Accelerated Parallel RK4 ODE Solver — Development Log

> Single documentation file tracking all implementation progress.
> Updated incrementally as each component is built and tested.

---

## Project Structure

```
final/
├── requirements.txt
├── DEVLOG.md                     ← this file
├── src/
│   ├── __init__.py
│   ├── ode_systems.py            ✅ Implemented
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── classical_rk4.py      ✅ Implemented
│   │   ├── parareal.py           ✅ Implemented
│   │   ├── neural_augmented_rk4.py  ⬜ Pending
│   │   └── gpu_engine.py            ⬜ Pending
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── coarse_propagator.py  ✅ Implemented
│   │   ├── k_factor_residual.py  ✅ Implemented
│   │   └── trust_gate.py         ✅ Implemented
│   ├── training/
│   │   ├── __init__.py
│   │   ├── data_generator.py     ✅ Implemented
│   │   ├── train_coarse.py       ✅ Implemented
│   │   ├── train_k_factor.py     ✅ Implemented
│   │   └── train_all.py          ✅ Implemented
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                 ⬜ Pending
├── benchmarks/                      ⬜ Pending
├── demo/                            ⬜ Pending
├── tests/
│   ├── test_rk4_correctness.py      ✅ 21 tests
│   ├── test_networks.py             ✅ 19 tests
│   └── test_parareal_convergence.py ✅  6 tests
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
- `rk4_step` static method exposes all 4 k-factors (k1, k2, k3, k4) — training data for k-factor residual network.
- `RK4StepResult` and `SolveResult` dataclasses for clean return types.
- `solve_batched` attempts `torch.vmap` first, falls back to sequential loop if unsupported.
- Input validation with clear error messages.
- Logging at INFO (high-level) and DEBUG (per-step) levels.

### 1.3 Tests (`tests/test_rk4_correctness.py`) — 21 tests ✅

- `TestODESystems`: Registry, dimensions, shapes, param_vector, param_ranges
- `TestClassicalRK4Basic`: Output shapes, IC preservation, k-factors, error handling
- `TestRK4VsAnalytical`: Position and velocity accuracy vs analytical solution
- `TestRK4VsScipy`: Cross-validation against scipy.integrate.solve_ivp (all 4 systems)
- `TestBatchedSolve`: Batched results match single-solve results
- `TestSolveInterval`: Interval endpoint matches full solve endpoint

---

## Phase 2 — Neural Components

### Branch: `final/phase2-neural-components`

### 2.1 Coarse Propagator (`src/networks/coarse_propagator.py`)

**Status**: ✅ Complete

Meta-propagator NN conditioned on θ_ODE for cross-problem generalisation.

**Architecture**:
- Input: `[y_n (D), t_n (1), delta_t (1), theta_ODE (P)]`
- Trunk: 3 hidden layers × 128 units, LayerNorm + SiLU
- Dual output heads:
  - **State head**: `y_hat_{n+1}` (dim D)
  - **Confidence head**: `epsilon_n ∈ [0, 1]` (sigmoid)
- Hidden dims aligned to multiples of 8 for GPU Tensor Core efficiency

### 2.2 k-Factor Residual Network (`src/networks/k_factor_residual.py`)

**Status**: ✅ Complete

Predicts corrections δ₂, δ₃, δ₄ so that `k_hat_i = k1 + delta_i` approximates true RK4 stages.

**Architecture**:
- Input: `[k1 (D), y_n (D), t_n (1), h (1)]`
- Projection → 2 ResidualBlocks (skip connections for gradient flow) → Output
- Output: 3 corrections, each of dimension D
- `predict_k_factors()` adds k1 back for direct use

**Key insight**: Learning *residuals* is easier than learning absolute values because δ₂, δ₃, δ₄ are small and smooth.

### 2.3 Trust Gate (`src/networks/trust_gate.py`)

**Status**: ✅ Complete

Adaptive gating mechanism for selective Parareal fine passes.

**Behaviour**:
- Per-slab decision: accept NN prediction if error_estimate < threshold
- Threshold decays geometrically: `tau = max(tau_0 * decay^k, tau_min)`
- Early iterations → conservative (most slabs corrected)
- Later iterations → permissive (more slabs trusted → faster convergence)
- `get_stats()` for dashboard monitoring

### 2.4 Data Generator (`src/training/data_generator.py`)

**Status**: ✅ Complete

Generates two types of training data from diverse RK4 trajectories:
1. **Coarse data**: `(y_n, t_n, delta_t, theta_ODE) → y_{n+1}` — sub-sampled from fine RK4
2. **k-factor data**: `(k1, y_n, t_n, h) → (k2, k3, k4)` — collected from every RK4 step

Both sample random ICs AND random ODE parameters for meta-generalisation.

### 2.5 Tests (`tests/test_networks.py`) — 19 tests ✅

- `TestCoarsePropagatorNet`: Shapes, confidence range [0,1], gradient flow, convenience method
- `TestKFactorResidualNet`: Shapes, residual structure, skip connections, gradient flow
- `TestTrustGate`: Threshold decay, gating decisions, floor, reset, stats
- `TestDataGenerator`: Shapes, device transfer, parameter diversity

---

## Phase 3 — Integration

### Branch: `final/phase3-integration`

### 3.1 Parareal Solver (`src/solvers/parareal.py`)

**Status**: ✅ Complete

Full parallel-in-time solver implementing the Parareal algorithm:

**Algorithm**:
1. **Partition**: Split `[t0, T]` into P equal sub-intervals
2. **Initialise**: Run NN coarse propagator sequentially for initial guesses
3. **Iterate** until convergence:
   - Fine pass: RK4 on all active slabs
   - Trust gate: decide which slabs to correct vs skip
   - Correction: `U_{n+1}^{k+1} = G(U_n^{k+1}) + [F(U_n^k) - G(U_n^k)]`
   - Convergence check: `max|U^{k+1} - U^k| < tolerance?`

**Key features**:
- `PararealResult` dataclass with full diagnostics (convergence history, fine solve counts, trust gate stats, wall time)
- Trust gate integration for adaptive slab skipping
- Comprehensive logging at each iteration

### 3.2 Training Pipelines (`src/training/train_coarse.py`, `train_k_factor.py`)

**Status**: ✅ Complete

End-to-end training for both neural networks:
- Data generation → train/val split → Adam + cosine LR → loss history
- Coarse trainer: MSE data loss (physics residual infrastructure ready)
- k-factor trainer: MSE on residual targets `(delta_i = k_i - k1)`

### 3.3 Training Orchestrator (`src/training/train_all.py`)

**Status**: ✅ Complete

- Trains both networks sequentially for a given ODE system
- Saves model weights to `trained_models/`
- CLI entry point: `py -3.11 -m src.training.train_all --system damped_oscillator`

### 3.4 Tests (`tests/test_parareal_convergence.py`) — 6 tests ✅

- Converges with untrained NN (algorithm correctness independent of coarse quality)
- Convergence error decreases across iterations
- Matches serial RK4 at slab boundaries
- IC preserved at first boundary
- Result diagnostics fully populated
- Trust gate stats populated when enabled

---

## Phase 4 — Demo & Polish

### Branch: `final/phase4-demo` (planned)

⬜ `demo/app.py` — Streamlit dashboard
⬜ `benchmarks/benchmark_solvers.py` — Full speedup analysis
⬜ `src/visualization/plots.py` — Plotting utilities
⬜ Final `README.md`

---

## Test Summary

| Phase | Test File | Tests | Status |
|-------|-----------|-------|--------|
| 1 | `test_rk4_correctness.py` | 21 | ✅ All pass |
| 2 | `test_networks.py` | 19 | ✅ All pass |
| 3 | `test_parareal_convergence.py` | 6 | ✅ All pass |
| **Total** | | **46** | **✅ All pass** |
