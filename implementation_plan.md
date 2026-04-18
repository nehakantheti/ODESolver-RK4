# Neural-Accelerated Parallel RK4 ODE Solver — Implementation Plan

## Goal

Build a working demo of a **GPU-native, neural network-augmented, parallel-in-time RK4 ODE solver** that combines four techniques into a single two-level hybrid pipeline:

1. **Parareal parallel-in-time decomposition** — split the time domain into P slabs solved concurrently
2. **Meta-propagator NN coarse solver** — a learned coarse propagator conditioned on ODE parameters for cross-problem generalization
3. **k-factor residual prediction** — a small residual network that predicts corrections to RK4 stages k₂, k₃, k₄ given an exact k₁, reducing 3 sequential `f` evaluations to 1 NN forward pass
4. **Adaptive trust gate** — per-slab confidence-based gating that selectively skips fine correction where the NN is already accurate

All stages (coarse NN + batched fine RK4 via `torch.vmap`) run on GPU, eliminating CPU↔GPU transfer bottlenecks.

---

## Research Foundation & Novelty

### Literature Landscape

| Existing Work | What They Do | Gap This Project Fills |
|---|---|---|
| Parareal (Lions et al., 2001) | PinT with cheap numerical coarse solver | Coarse solver is still a numerical integrator (e.g., Euler) |
| Ibrahim et al. — PINN-Parareal ([arXiv:2303.03848](https://arxiv.org/abs/2303.03848)) | PINN as coarse propagator; GPU-CPU heterogeneous | PINN training is expensive; fixed to one ODE instance |
| RandNet-Parareal (NeurIPS 2024) | Random NN learns F−G discrepancy; ×125 speedup | Random weights lack structured physics inductive bias |
| Othmane & Flaßkamp ([arXiv:2504.05493](https://arxiv.org/abs/2504.05493), 2025) | Additive NN correction for RK; embedded RK safety | Correction only; no parallel-in-time |
| Dherin et al. (Google, 2025) | RK methods as deep learning optimizers; DAL adaptive stepping | Optimization framing, not forward ODE simulation |
| Agrawal et al. (RK4 + NN for airbrakes) | NN replaces RK4 entirely | Loses interpretability; not a hybrid |

### 4 Novel Contributions of This Project

1. **Meta-propagator**: The coarse NN is conditioned on ODE parameters θ_ODE (damping, frequency, etc.), not just state — enabling **cross-problem generalization** unlike prior PINN-Parareal or RandNet-Parareal work which train per-problem.

2. **k-factor residual prediction**: Instead of replacing the solver, a lightweight network learns **corrections** δ₂, δ₃, δ₄ to approximate k₂, k₃, k₄ given an exact k₁. This preserves RK4's accuracy while collapsing 3 sequential `f` evaluations into 1 parallel NN forward pass. Gated: only activated when `f` is computationally expensive.

3. **Trust-gated Parareal**: At each Parareal iteration, a per-slab confidence score from the NN decides whether to accept the NN prediction directly or trigger a fine solver correction — reducing iteration count and total compute. Inspired by adaptive step-size DAL from the Google RK paper.

4. **Fully GPU-native pipeline**: Both coarse (NN) and fine (batched RK4 via `torch.vmap`) stages run on the same GPU. No CPU↔GPU transfers during the Parareal loop — a key bottleneck in prior CPU-based coarse solver implementations.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: ODE Problem                    │
│         f(t, y),  y₀,  [t_start, t_end],  params       │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │   Time Domain Decomposer    │
          │  Split [t₀, T] into P slabs │
          └──────────────┬──────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   [Slab 0]         [Slab 1]  ...   [Slab P-1]
        │                │                │
        └────────────────┼────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     COARSE PASS (Serial)    │
          │   Meta-NN Coarse Propagator │
          │   G(yₙ, tₙ, ΔT, θ_ODE)     │
          │     → ŷ_{n+1}, εₙ           │
          │   [runs on GPU, very fast]  │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │   TRUST GATE (per slab)     │
          │  εₙ < τ? → Accept NN       │  ← Novel: skip fine where NN
          │  εₙ ≥ τ? → Run fine RK4    │     is already confident
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     FINE PASS (Parallel)    │
          │  Batched RK4 via vmap       │
          │  [all active slabs on GPU   │
          │   in parallel, one kernel]  │
          │                             │
          │  (Optional: k-factor net    │
          │   accelerates each RK4      │
          │   step when f is expensive) │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │   Parareal Correction       │
          │  y^{k+1}_{n+1} =            │
          │    G(y^{k+1}_n) +            │
          │    [F(y^k_n) − G(y^k_n)]    │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Convergence Check          │
          │  ‖y^{k+1} − y^k‖ < tol?    │
          │  Yes → Output y(t)          │
          │  No  → Next iteration k+1   │
          └─────────────────────────────┘
```

---

## Proposed Changes — File Structure

All code lives in `final/`:

```
final/
├── README.md                        # Setup, usage, and demo guide
├── requirements.txt                 # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── ode_systems.py               # 4 benchmark ODE system definitions
│   │
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── classical_rk4.py         # Pure PyTorch RK4 (CPU + GPU + batched via vmap)
│   │   ├── neural_augmented_rk4.py  # RK4 with k-factor residual prediction
│   │   ├── parareal.py              # Full Parareal loop with pluggable coarse/fine
│   │   └── gpu_engine.py            # GPU optimization: AMP, torch.compile, vmap batching
│   │
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── coarse_propagator.py     # Meta-propagator NN (conditioned on θ_ODE)
│   │   ├── k_factor_residual.py     # Residual net: k₁ → δ₂, δ₃, δ₄ corrections
│   │   └── trust_gate.py            # Confidence head + gating logic
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── data_generator.py        # Generate training data from classical RK4
│   │   ├── train_coarse.py          # Train the meta-propagator
│   │   ├── train_k_factor.py        # Train the k-factor residual network
│   │   └── train_all.py             # End-to-end training orchestrator
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py                 # Static Matplotlib + Plotly plots
│       └── animation.py             # Animated Parareal convergence
│
├── cpp_baseline/
│   ├── rk4_baseline.cpp             # C++ ground truth RK4 (cleaned from mid/)
│   └── Makefile
│
├── benchmarks/
│   ├── benchmark_solvers.py         # Full cross-solver benchmark
│   └── benchmark_gpu.py             # GPU-specific: AMP, compile, vmap, batch size sweep
│
├── demo/
│   └── app.py                       # Streamlit interactive dashboard
│
├── tests/
│   ├── test_rk4_correctness.py      # RK4 vs analytical / scipy
│   ├── test_networks.py             # Network shape, range, gradient flow
│   ├── test_parareal_convergence.py # Convergence to serial fine solution
│   └── test_trust_gate.py           # Calibration and safety tests
│
└── trained_models/                  # Saved .pt weights (generated by training)
    └── .gitkeep
```

---

## Component Details

### Component 1 — ODE Systems (`ode_systems.py`)

Four benchmark systems of increasing difficulty:

| # | System | Equations | Purpose |
|---|--------|-----------|---------|
| 1 | **Damped Harmonic Oscillator** | `x'' + cx' + kx = 0` | Continuity with `mid/`; linear, analytical solution exists |
| 2 | **Lotka-Volterra** | `x' = αx − βxy`, `y' = δxy − γy` | Nonlinear, oscillatory, conservation test |
| 3 | **Van der Pol Oscillator** | `x'' − μ(1−x²)x' + x = 0` | Limit cycles; stiff for large μ |
| 4 | **Lorenz Attractor** | `x'=σ(y−x)`, `y'=x(ρ−z)−y`, `z'=xy−βz` | Chaotic — stress test for both NN and Parareal |

Each system is a callable class exposing:
```python
class ODESystem:
    name: str
    dim: int                          # state dimension
    param_names: list[str]            # e.g., ["mass", "damping", "stiffness"]
    default_params: dict
    default_ic: Tensor                # initial condition
    default_t_span: tuple[float, float]

    def f(self, t, y, params) -> Tensor:   # the derivative function
    def analytical(self, t, params) -> Tensor | None:  # if available
```

> [!NOTE]
> The `params` argument is critical — it flows into the meta-propagator as θ_ODE, enabling cross-parameter generalization.

---

### Component 2 — Classical RK4 Solver (`classical_rk4.py`)

#### [NEW] `classical_rk4.py`

Pure PyTorch RK4 with three execution modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `solve_single` | One trajectory, one IC | Ground truth generation |
| `solve_batched` | Many ICs simultaneously via `torch.vmap` | Parareal fine pass; ensemble runs |
| `solve_gpu` | `solve_batched` + AMP + `torch.compile` | Maximum throughput |

**Mathematical formulation** (standard RK4):
```
k₁ = h · f(tₙ, yₙ)
k₂ = h · f(tₙ + h/2, yₙ + k₁/2)
k₃ = h · f(tₙ + h/2, yₙ + k₂/2)
k₄ = h · f(tₙ + h, yₙ + k₃)
yₙ₊₁ = yₙ + (1/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

**Key design**: The `solve_batched` function uses `torch.vmap` to vectorize across initial conditions. This means all P time slabs in the Parareal fine pass execute as **one GPU kernel launch** — no Python loop over slabs.

```python
# Pseudocode for vmap-based batched solve
def rk4_step(y, t, h, f):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Vectorize across batch dimension (different ICs per slab)
batched_rk4_step = torch.vmap(rk4_step, in_dims=(0, None, None, None))
```

---

### Component 3 — Meta-Propagator NN Coarse Solver (`coarse_propagator.py`)

#### [NEW] `coarse_propagator.py`

**Purpose**: Given state yₙ at time tₙ, quickly predict yₙ₊₁ at tₙ₊₁ over a coarse timestep ΔT. Conditioned on ODE parameters θ_ODE for cross-problem generalization.

**Architecture** — Small MLP with parameter conditioning:
```
Input: [yₙ (dim D), tₙ, ΔT, θ_ODE (dim P)] → dim D+2+P
    ↓
Linear(D+2+P, 128) → LayerNorm → SiLU
    ↓
Linear(128, 128) → LayerNorm → SiLU
    ↓
Linear(128, 128) → LayerNorm → SiLU
    ↓
├── Linear(128, D)     → ŷ_{n+1}     (state prediction)
└── Linear(128, 1)     → sigmoid → εₙ (confidence score)
```

**Training loss** — Semi-physics-informed (data + physics residual):

```
L = ‖ŷ − y_fine‖²  +  λ · ‖ŷ' − f(t, ŷ)‖²
    ⎣_data loss_⎦     ⎣__physics residual__⎦
```

The physics residual ensures the NN doesn't just memorize data but learns dynamically consistent trajectories — critical for generalization to unseen parameters.

**Confidence head training** — Expected Calibration Error (ECE):
```
L_conf = Σ_bins |accuracy(bin) − confidence(bin)|
```
This ensures the confidence output εₙ is **calibrated**: when the network says 0.95 confidence, it should be correct ~95% of the time.

**Why meta-propagator beats prior work**:
- PINN-Parareal (Ibrahim et al.) trains a separate PINN per ODE instance → expensive
- RandNet-Parareal uses random weights → no physics inductive bias
- This meta-propagator trains once on a **family** of ODEs (varying damping, frequency, etc.) and generalizes at inference time to unseen parameter combinations

---

### Component 4 — k-Factor Residual Network (`k_factor_residual.py`)

#### [NEW] `k_factor_residual.py`

**Purpose**: Accelerate individual RK4 steps by replacing 3 sequential `f` evaluations with 1 NN forward pass.

**Key insight**: k₁ is always computed exactly (requires one call to `f`). Given k₁, the subsequent stages k₂, k₃, k₄ are correlated — a residual network can learn the corrections δ₂, δ₃, δ₄:

```
k₁ = h · f(tₙ, yₙ)                    ← exact, always computed
k̂₂ = k₁ + δ₂(k₁, yₙ, tₙ, h)          ← NN correction
k̂₃ = k₁ + δ₃(k₁, yₙ, tₙ, h)          ← NN correction
k̂₄ = k₁ + δ₄(k₁, yₙ, tₙ, h)          ← NN correction
```

**Architecture** — Residual MLP:
```
Input: [k₁ (dim D), yₙ (dim D), tₙ, h] → dim 2D+2
    ↓
Linear(2D+2, 96) → LayerNorm → SiLU
    ↓
Linear(96, 96) → LayerNorm → SiLU  (+ residual connection)
    ↓
Linear(96, 3*D) → Reshape to (3, D)
    ↓
Output: [δ₂, δ₃, δ₄] each of dim D
Final: k̂ᵢ = k₁ + δᵢ for i=2,3,4
```

**Training loss**:
```
L_k = ‖k̂₂ − k₂‖² + ‖k̂₃ − k₃‖² + ‖k̂₄ − k₄‖²
```

Where k₂, k₃, k₄ are ground truth from classical RK4.

**Use-case gating**: This component is **optional per time slab**. A simple heuristic decides activation:
- If `f` evaluation time > NN forward pass time → activate k-factor net
- If `f` is cheap (e.g., simple polynomial ODE) → bypass, use classical RK4
- For the demo: manually toggle via a flag; show speedup difference

> [!IMPORTANT]
> **Why residual prediction, not direct prediction**: Residual learning (predicting δ = k̂ − k₁) is easier than predicting absolute values. The network only needs to learn the *difference* between stages — which is often small and smooth. This gives faster convergence and better generalization.

---

### Component 5 — Adaptive Trust Gate (`trust_gate.py`)

#### [NEW] `trust_gate.py`

**Purpose**: At each Parareal iteration, selectively skip the expensive fine RK4 correction on slabs where the NN coarse propagator is already accurate.

**Decision rule per slab n**:
```
action(n) = {
    accept ŷ_{n+1}^NN      if εₙ < τ    (high confidence)
    run fine RK4 correction  if εₙ ≥ τ    (low confidence)
}
```

**Threshold τ**:
- Option A: Fixed hyperparameter (e.g., τ = 0.05), tuned on validation set
- Option B: Learnable — a small network that outputs τ based on the Parareal iteration number k and overall convergence rate
- Default: Option A for simplicity (Option B as stretch goal)

**Effect**: In later Parareal iterations, the NN gets better (it's been corrected by fine solves). More slabs pass the trust gate → fewer fine solves needed → faster convergence.

**Safety**: If the trust gate accepts a bad prediction, the next Parareal iteration's convergence check catches it (the error will remain above tolerance). The gate accelerates convergence but cannot cause *divergence* — Parareal's correction formula inherently damps errors.

---

### Component 6 — Parareal Solver (`parareal.py`)

#### [NEW] `parareal.py`

**The full Parareal algorithm with neural coarse propagator and trust gating**:

```
INPUTS:
  f       — ODE derivative function
  y₀      — initial condition
  [t₀, T] — time interval
  P       — number of parallel slabs
  tol     — convergence tolerance
  G       — neural coarse propagator (meta-propagator)
  F       — fine solver (classical or k-factor-augmented RK4)

ALGORITHM:
  1. Partition: T_n = t₀ + n·ΔT  for n = 0,...,P  where ΔT = (T−t₀)/P

  2. INITIALIZE (serial, on GPU):
     U₀⁰ = y₀
     For n = 0,...,P−1:
       U_{n+1}⁰, ε_n = G(U_n⁰, T_n, ΔT, θ_ODE)

  3. ITERATE k = 0, 1, 2, ...:

     a. FINE PASS (parallel, on GPU via vmap):
        For ALL slabs n = 0,...,P−1 IN PARALLEL:
          If trust_gate says "skip slab n": continue
          F_n^k = F(U_n^k, T_n, T_{n+1})   # fine RK4 over slab

     b. COARSE CORRECTION (serial, on GPU):
        For n = 0,...,P−1:
          G_new = G(U_n^{k+1}, T_n, ΔT, θ_ODE)
          G_old = G(U_n^k,     T_n, ΔT, θ_ODE)   # cached from prev
          U_{n+1}^{k+1} = G_new + [F_n^k − G_old]

     c. CONVERGENCE CHECK:
        If max_n ‖U_n^{k+1} − U_n^k‖ < tol → STOP

  4. RETURN: U_n^K at all time points
```

**Speedup analysis**:
- Serial RK4 cost: N_steps × cost(f)
- Parareal cost: K_iter × [cost(G) × P + cost(F) × ΔT/h]
- With NN coarse: cost(G) is tiny (one NN forward pass)
- With vmap fine: cost(F) is amortized across P slabs on GPU
- With trust gate: some fine passes skipped entirely
- **Expected speedup**: 5-20× for smooth ODEs, 2-5× for chaotic systems

---

### Component 7 — GPU Engine (`gpu_engine.py`)

#### [NEW] `gpu_engine.py`

Centralizes all GPU optimization strategies:

| Strategy | Implementation | When Used |
|----------|---------------|-----------|
| **Mixed Precision** | `torch.amp.autocast` for NN inference; FP32 accumulation for RK4 state | Always for NN; configurable for RK4 |
| **torch.compile** | JIT-compile the `rk4_step` and NN forward functions | After warmup |
| **vmap batching** | `torch.vmap` for parallel slab solving | Parareal fine pass |
| **Gradient checkpointing** | `torch.utils.checkpoint` during NN training | Long trajectories |
| **Memory management** | `optimizer.zero_grad(set_to_none=True)`, layer width alignment to 8 | Training loop |

---

### Component 8 — Training Pipeline (`training/`)

#### [NEW] `data_generator.py`

Generates training data by running high-accuracy classical RK4:

```python
def generate_coarse_data(ode_system, n_trajectories, param_ranges):
    """
    For each trajectory:
      1. Sample random IC from distribution around default
      2. Sample random ODE params from param_ranges
      3. Run RK4 with small h (high accuracy)
      4. Store (y_n, t_n, ΔT, θ_ODE) → y_{n+1} pairs
    Returns: dataset of (input, target) pairs
    """

def generate_k_factor_data(ode_system, n_steps, param_ranges):
    """
    For each RK4 step:
      1. Compute k₁, k₂, k₃, k₄ exactly
      2. Store (k₁, y_n, t_n, h) → (k₂, k₃, k₄) pairs
    Returns: dataset of (input, target) pairs
    """
```

**Data diversity**: Sample initial conditions AND ODE parameters from ranges to ensure the meta-propagator generalizes:
- Damped oscillator: damping c ∈ [0.01, 0.5], spring k ∈ [0.5, 2.0]
- Lotka-Volterra: α ∈ [0.5, 2.0], β ∈ [0.01, 0.1]
- Van der Pol: μ ∈ [0.1, 5.0]
- Lorenz: σ ∈ [8, 12], ρ ∈ [25, 30], β ∈ [2, 3.5]

#### [NEW] `train_coarse.py`

| Setting | Value |
|---------|-------|
| Optimizer | Adam, lr=1e-3 with cosine annealing |
| Epochs | ~5000 |
| Batch size | 256 |
| Loss | MSE + λ·physics_residual (λ=0.1) + confidence calibration |
| Validation | 20% held-out trajectories |

#### [NEW] `train_k_factor.py`

| Setting | Value |
|---------|-------|
| Optimizer | Adam, lr=1e-3 |
| Epochs | ~3000 |
| Batch size | 512 |
| Loss | MSE on (δ₂, δ₃, δ₄) |
| Validation | 20% held-out steps |

#### [NEW] `train_all.py`

Orchestrates end-to-end:
1. Generate data for all 4 ODE systems
2. Train coarse propagator (with confidence head)
3. Train k-factor residual network
4. Validate both on held-out data
5. Save models to `trained_models/`
6. Print summary: training time, final loss, validation metrics

---

## Demo Design — 4 Scenarios

### Demo 1: Correctness Verification
- Solve Van der Pol oscillator with all 3 solvers
- Overlay plot: serial RK4 vs Neural-Augmented RK4 vs Neural-Parareal → visually indistinguishable
- Error plot: |y_parareal − y_serial| over time
- Parareal convergence curve: error vs iteration number k

### Demo 2: Speedup Benchmark
- Sweep number of time slabs P from 1 to 64
- Bar chart: wall-clock time for serial RK4 vs Parareal (Euler coarse) vs Parareal (NN coarse, ours)
- Expected result: NN coarse converges in fewer Parareal iterations → better speedup

### Demo 3: GPU Utilization
- Show GPU memory and compute utilization during the fine parallel pass
- Demonstrate that vmap-batched fine solver saturates GPU cores vs sequential per-slab integration
- Scaling curve: throughput (trajectories/sec) vs batch size

### Demo 4: Generalization
- Train meta-propagator on harmonic oscillator family (c ∈ [0.05, 0.3])
- Test on unseen damping coefficient (c = 0.4, outside training range)
- Show the NN coarse propagator still gives good initialization → fast Parareal convergence even out-of-distribution

---

## Dashboard (`demo/app.py`)

**Streamlit-based** interactive dashboard with 4 tabs matching the demos above:

**Tab 1 — Solver Comparison**: ODE selector, parameter sliders, animated side-by-side trajectories, error over time
**Tab 2 — Parareal Visualization**: Animated iteration-by-iteration convergence, trust gate activation map, iteration count & speedup
**Tab 3 — GPU Benchmark**: Bar charts (CPU vs GPU, baseline vs AMP vs compiled vs batched), scaling curves, memory usage
**Tab 4 — Neural Network Inspector**: k-factor predictions vs ground truth, confidence calibration plot, error corrector magnitudes, training loss curves

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Solution error | ‖y_parareal − y_reference‖₂ | < 1e-5 (matches serial RK4) |
| Parareal iterations to convergence | Fewer = better coarse propagator | < P/3 for smooth ODEs |
| Wall-clock speedup | vs serial fine RK4 | > 5× for P=16 on smooth ODEs |
| GPU utilization % | During fine parallel pass | > 70% with vmap |
| Coarse propagator accuracy | ‖ŷ − y_fine‖ per slab | < 1e-3 after training |
| Trust gate efficiency | % of slabs where fine correction was skipped | > 30% in later iterations |
| k-factor prediction MSE | ‖k̂ᵢ − kᵢ‖² | < 1e-6 |

---

## Dependencies

```
torch >= 2.1.0           # Core framework; torch.compile + vmap support
numpy >= 1.24
matplotlib >= 3.7
plotly >= 5.15            # Interactive plots for dashboard
streamlit >= 1.28         # Dashboard framework
pandas >= 2.0             # Data handling for benchmarks
tqdm >= 4.65              # Training progress bars
scipy >= 1.11             # Reference solutions for validation
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NN coarse propagator diverges on chaotic ODEs (Lorenz) | Medium | High | Shorter coarse timestep ΔT; increase physics loss weight λ; fall back to Euler coarse |
| k-factor net overhead exceeds savings for simple `f` | High for simple ODEs | Low | Use-case gating: only enable when `f` is computationally expensive |
| Parareal speedup limited by serial coarse pass | Medium | Medium | NN on GPU makes coarse pass extremely fast; speedup bound depends on iteration count K |
| Trust gate miscalibrated — skips needed fine corrections | Low-Medium | Medium | Validate calibration on held-out trajectories (ECE < 0.05) before enabling; conservative default τ |
| `torch.vmap` limitations on Windows / older PyTorch | Low | High | Fallback to manual batching with `torch.stack` if vmap unsupported |
| Training data insufficient for meta-propagator generalization | Low | Medium | Large parameter sweep + data augmentation during generation |

---

## Phased Roadmap

### Phase 1 — Foundation
- Set up project structure, `requirements.txt`, `__init__.py` files
- Implement `ode_systems.py` with all 4 benchmark systems
- Implement `classical_rk4.py` (single, batched via vmap, GPU)
- Port and clean `rk4_baseline.cpp` from `mid/phase1`
- Write correctness tests vs analytical solutions and scipy

### Phase 2 — Neural Components
- Implement `data_generator.py` — generate diverse training data
- Implement `coarse_propagator.py` architecture + `train_coarse.py`
- Implement `k_factor_residual.py` architecture + `train_k_factor.py`
- Implement `trust_gate.py` with calibration loss
- Validate NN outputs: shapes, ranges, gradient flow

### Phase 3 — Integration
- Implement `neural_augmented_rk4.py` — RK4 with k-factor gating
- Implement `parareal.py` — full algorithm with NN coarse + trust gate
- Implement `gpu_engine.py` — AMP, compile, vmap wrappers
- Wire everything together: `train_all.py` orchestrator
- Run full pipeline end-to-end on all 4 ODE systems

### Phase 4 — Demo & Polish
- Build `benchmark_solvers.py` — full speedup analysis
- Build `benchmark_gpu.py` — GPU utilization profiling
- Build Streamlit `app.py` with all 4 tabs
- Create animated Parareal convergence visualization
- End-to-end integration testing
- Write `README.md` with setup guide + demo walkthrough

---

## Verification Plan

### Automated Tests

1. **`test_rk4_correctness.py`**:
   - Classical RK4 vs analytical (damped oscillator): error < 1e-6
   - Classical RK4 vs `scipy.integrate.solve_ivp` for all 4 systems: relative error < 1e-5
   - Batched RK4 output matches single-solve output identically

2. **`test_networks.py`**:
   - Coarse propagator output shape: (batch, D)
   - Confidence output range: [0, 1] (sigmoid)
   - k-factor residual output shape: (batch, 3, D)
   - All networks have functioning gradient flow (no NaN/Inf)

3. **`test_parareal_convergence.py`**:
   - Parareal converges to serial fine solution within tolerance
   - Iterations K < P (otherwise no speedup possible)
   - Convergence verified for all 4 benchmark systems

4. **`test_trust_gate.py`**:
   - Calibration: ECE < 0.05 on held-out data
   - Safety: enabling trust gate does not increase solution error beyond 2× tolerance
   - Gate activates more in later Parareal iterations (expected behavior)

### Manual Verification
- Run Streamlit dashboard, visually verify trajectory overlaps
- Confirm Parareal animation shows progressive convergence
- Verify GPU benchmark charts show expected speedup trends
- Profile with `torch.profiler` to confirm vmap kernel fusion

### Benchmark Runs
```bash
python benchmarks/benchmark_solvers.py --systems all --slabs 1,2,4,8,16,32,64
python benchmarks/benchmark_gpu.py --batch_sizes 1,10,100,1000
```
