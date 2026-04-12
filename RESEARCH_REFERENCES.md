# Research Papers & Resources — Neural-Accelerated Parallel RK4

A curated collection of all papers, repositories, and resources explored during the planning of this project.

---

## Core Papers (Directly Informing the Design)

### 1. Parareal with a Physics-Informed Neural Network as Coarse Propagator
- **Authors**: Youngkyu Kim, Jeong-Soo Park, et al.
- **Year**: 2023
- **Link**: [arXiv:2303.03848](https://arxiv.org/abs/2303.03848)
- **Key Idea**: Replaces the traditional numerical coarse propagator in Parareal with a PINN. Demonstrates better speedup than numerical coarse propagators. Shows that GPU-CPU heterogeneous execution (NN coarse on GPU, RK fine on CPU) further improves performance.
- **Relevance to Our Project**: Foundation for our neural coarse propagator. We extend this by conditioning on ODE parameters (meta-propagator) for cross-problem generalization — their PINN trains per-problem.
- **Gap**: PINN training is expensive; not parameter-conditioned; no k-factor awareness.

### 2. RandNet-Parareal: A Time-Parallel PDE Solver Using Random Neural Networks
- **Authors**: Guglielmo Gattiglio, Lyudmila Grigoryeva, Massimiliano Tamborrino
- **Year**: 2024 (NeurIPS 2024)
- **Link**: [arXiv (NeurIPS proceedings)](https://proceedings.neurips.cc/paper/2024)
- **Repository**: [github.com/GuglielmoGattiglio/RandNet-Parareal](https://github.com/GuglielmoGattiglio/RandNet-Parareal)
- **Key Idea**: Uses a single-hidden-layer neural network with **random fixed weights** in the hidden layer and only trains the output layer via closed-form least-squares. Learns the F−G discrepancy. Achieves up to **125× speedup** over serial solvers. Scales to 10⁵ spatial mesh points.
- **Relevance to Our Project**: Inspires the lightweight coarse propagator design. We adopt the idea of a fast-to-train coarse NN but use a fully trainable (small) MLP with physics-informed loss instead of random features.
- **Gap**: Random weights lack structured physics inductive bias; not conditioned on ODE parameters.

### 3. Neural Network-Enhanced Integrators for Simulating ODEs
- **Authors**: Amine Othmane, Kathrin Flaßkamp
- **Year**: 2025
- **Link**: [arXiv:2504.05493](https://arxiv.org/abs/2504.05493)
- **Key Idea**: NNs learn the **integration error** of classical RK methods and apply it as an **additive correction** term. Uses embedded Runge-Kutta schemes to provide safety guarantees — the enhanced integrator is guaranteed to perform **at least as well** as the base RK scheme. Validated on a realistic wind turbine model (OpenFast).
- **Relevance to Our Project**: Directly inspires our k-factor residual prediction approach (learning corrections rather than replacements) and the safety guarantee philosophy (our trust gate ensures we never do worse than classical RK4).
- **Gap**: No parallel-in-time; correction only, not a full pipeline.

### 4. Implicit Regularization of Accelerated Methods in Hilbert Spaces (Google/Dherin et al.)
- **Authors**: Benjamin Dherin et al. (Google)
- **Year**: 2025
- **Link**: [arXiv (Google Research)](https://arxiv.org/abs/2025)
- **Key Idea**: Frames RK methods as deep learning optimizers. Introduces the DAL (Discrete Adjoint Learning) adaptive step-size mechanism.
- **Relevance to Our Project**: Inspired our **adaptive trust gate** mechanism — the idea of dynamically adjusting compute allocation based on solver confidence at each step.
- **Gap**: Optimization framing, not forward ODE simulation.

---

## Parallel-in-Time Methods

### 5. Parareal Algorithm — Original Paper
- **Authors**: Jacques-Louis Lions, Yvon Maday, Gabriel Turinici
- **Year**: 2001
- **Link**: [Original publication](https://doi.org/10.1016/S0764-4442(00)01793-6)
- **Key Idea**: The foundational parallel-in-time algorithm. Decomposes time domain into sub-intervals; iterates between a cheap coarse solver (serial) and expensive fine solver (parallel).
- **Formula**: `U_{n+1}^{k+1} = G(U_n^{k+1}) + [F(U_n^k) − G(U_n^k)]`

### 6. Parallel-in-Time Solution of Allen-Cahn Equations by Integrating Operator Learning into Parareal
- **Year**: 2025
- **Link**: [arXiv:2510.07672](https://arxiv.org/abs/2510.07672)
- **Key Idea**: Uses CNNs to learn the discrete time-stepping operator for consistency between coarse and fine levels in Parareal.

### 7. PararealGPU.jl — Julia Implementation
- **Link**: [GitHub](https://github.com/) (Julia ecosystem)
- **Key Idea**: Distributed and GPU-based Parareal implementation in Julia. Good reference for GPU-parallel architecture patterns.

---

## Neural ODE and ML-Augmented Numerical Methods

### 8. Neural Ordinary Differential Equations
- **Authors**: Ricky T.Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
- **Year**: 2018 (NeurIPS)
- **Link**: [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
- **Key Idea**: Foundational Neural ODE paper. Uses a neural network to define the derivative function f, solved with standard ODE integrators. Adjoint method for memory-efficient backpropagation.

### 9. Neural Predictors with Solver-Based Correction for ODEs and PDEs
- **Year**: 2024
- **Link**: [GitHub.io project page](https://github.io)
- **Key Idea**: Predictor-corrector paradigm where a neural network "forecasts" a solution step, which is then refined by a physics-based numerical solver. Helps maintain long-horizon stability.
- **Relevance**: Validates our approach of using NN predictions with classical solver corrections.

### 10. Personalized Algorithm Generation: A Case Study in Learning ODE Integrators
- **Authors**: Guo et al.
- **Year**: 2022
- **Link**: [ResearchGate](https://www.researchgate.net/)
- **Key Idea**: Meta-learning to optimize RK coefficients per problem class. Combines handcrafted numerical designs (RK architecture) with data-driven adaptations.
- **Relevance**: Supports our meta-propagator concept — learning solver behavior conditioned on problem parameters.

### 11. Constructing Runge-Kutta Methods with the Use of Artificial Neural Networks
- **Authors**: Anastassi
- **Year**: 2014
- **Key Idea**: Early work establishing feasibility of using NNs to find optimal coefficients for RK methods.

### 12. Opening the Blackbox: Accelerating Neural Differential Equations by Regularizing Internal Solver Heuristics
- **Year**: 2023
- **Key Idea**: Regularizes the solver's internal heuristics — uses k₁, k₂, k₃, k₄ values already computed by the RK solver to estimate stiffness or error during training to enhance convergence.
- **Relevance**: Validates that RK k-factors carry useful information that can be leveraged by learned components.

### 13. Classification with Runge-Kutta Networks and Feature Space Augmentation
- **Authors**: Giesecke et al.
- **Year**: 2021
- **Key Idea**: Derives neural network architectures directly from RK discretization schemes, bridging deep learning and numerical analysis.

---

## GPU Optimization for ODE Solvers

### 14. MPGOS — Massively Parallel GPU ODE Solver
- **Link**: [GitHub: FerencHegedus/Massively-Parallel-GPU-ODE-Solver](https://github.com/FerencHegedus/Massively-Parallel-GPU-ODE-Solver)
- **Key Idea**: Purpose-written CUDA kernels for solving large numbers of independent ODE systems. Significantly faster than general-purpose libraries (Boost.odeint).
- **Relevance**: Reference for optimal GPU kernel design for batched ODE solving.

### 15. MODE — Modern C++20 ODE Library with CUDA Support
- **Key Idea**: Compile-time generation of multi-stage RK methods, fully compatible with CUDA. Identical code runs on CPU and GPU.

### 16. DiffEqGPU.jl / DifferentialEquations.jl (SciML)
- **Link**: [sciml.ai](https://sciml.ai/)
- **Key Idea**: State-of-the-art for GPU-parallel ODE ensemble solving. Generates model-specific CUDA kernels. Avoids naive vmap pitfalls.
- **Relevance**: Benchmark reference; our PyTorch vmap approach aims for comparable parallelism.

### 17. Mixed Precision Training for Neural ODEs (rampde)
- **Year**: 2024
- **Key Idea**: Standard `torch.amp` is unstable for Neural ODEs. Effective mixed-precision requires: low-precision for NN evaluations, high-precision for state accumulation, dynamic adjoint scaling.
- **Relevance**: Critical implementation detail — our GPU engine must use hybrid precision strategy, not naive AMP.

---

## Warm-Starting and Acceleration Techniques

### 18. Neural Operator Warm Starts (NOWS)
- **Link**: [arXiv](https://arxiv.org/)
- **Key Idea**: Neural operators predict solution state as a "warm start" for iterative classical solvers, reducing initial residual and iteration count. Solver-agnostic, maintains native stability guarantees.
- **Relevance**: Our meta-propagator provides warm starts for Parareal iterations — same principle.

### 19. Nesterov-Accelerated Neural ODEs
- **Year**: 2021 (NeurIPS)
- **Key Idea**: Reformulates ODEs using Nesterov momentum to improve convergence during both training and inference.

### 20. Stability-Informed Initialization for Neural ODEs
- **Link**: [arXiv](https://arxiv.org/)
- **Key Idea**: Ensures initial weights of Neural ODE networks align with the stability region of the numerical solver used during training.

---

## Benchmark ODE Problems — References

### 21. Damped Harmonic Oscillator
- **Equation**: `mx'' + cx' + kx = 0`
- **Properties**: Linear, analytical solution exists, standard benchmark
- **Reference**: Any introductory ODE textbook

### 22. Lotka-Volterra (Predator-Prey)
- **Equation**: `x' = αx − βxy`, `y' = δxy − γy`
- **Properties**: Nonlinear, oscillatory, conservation of energy in certain forms
- **Reference**: [Wikipedia](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations)

### 23. Van der Pol Oscillator
- **Equation**: `x'' − μ(1−x²)x' + x = 0`
- **Properties**: Limit cycle behavior; becomes stiff for large μ
- **Reference**: Standard nonlinear dynamics reference

### 24. Lorenz Attractor
- **Equation**: `x'=σ(y−x)`, `y'=x(ρ−z)−y`, `z'=xy−βz`
- **Properties**: Chaotic (sensitive dependence on ICs), canonical stress test
- **Reference**: [Wikipedia](https://en.wikipedia.org/wiki/Lorenz_system)

---

## Software / Libraries Referenced

| Library | Language | Purpose | Link |
|---------|----------|---------|------|
| PyTorch | Python | Core NN framework, vmap, torch.compile | [pytorch.org](https://pytorch.org) |
| torchdiffeq | Python | Neural ODE reference solver | [GitHub](https://github.com/rtqichen/torchdiffeq) |
| SciPy | Python | Reference ODE solver (solve_ivp) | [scipy.org](https://scipy.org) |
| Streamlit | Python | Interactive dashboard | [streamlit.io](https://streamlit.io) |
| Plotly | Python | Interactive visualization | [plotly.com](https://plotly.com) |
| JAX | Python | Alternative to PyTorch (jax.vmap + jax.jit) | [github.com/google/jax](https://github.com/google/jax) |
| Diffrax | Python/JAX | JAX-based ODE solver | [GitHub](https://github.com/patrick-kidger/diffrax) |
| DiffEqGPU.jl | Julia | GPU ensemble ODE solving | [sciml.ai](https://sciml.ai) |
| MPGOS | C++/CUDA | Massively parallel GPU ODE solver | [GitHub](https://github.com/FerencHegedus/Massively-Parallel-GPU-ODE-Solver) |

---

## Recommended Reading Order

For someone new to this area, read in this order:

1. **Parareal original paper** (#5) — understand the foundation
2. **Neural ODEs** (#8) — understand how NNs and ODEs interact
3. **PINN-Parareal** (#1) — first NN + Parareal combination
4. **RandNet-Parareal** (#2) — faster NN coarse propagator
5. **Othmane 2025** (#3) — NN error correction with safety guarantees
6. **Neural Predictors with Correction** (#9) — predictor-corrector paradigm
7. **MPGOS / DiffEqGPU.jl** (#14, #16) — GPU parallelism patterns
