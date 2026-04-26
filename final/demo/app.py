"""Streamlit dashboard for the Neural-Accelerated Parallel RK4 Solver.

A 4-tab interactive demo showcasing:
    Tab 1 — Correctness: Classical RK4 vs analytical solutions
    Tab 2 — Neural Solver: Train + run Parareal with neural coarse propagator
    Tab 3 — Convergence: Animated Parareal convergence analysis
    Tab 4 — Benchmarks: Performance comparison and speedup analysis

Launch with:
    cd final
    py -3.11 -m streamlit run demo/app.py
"""

from __future__ import annotations

import sys
import os
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.ode_systems import get_system, SYSTEM_REGISTRY
from src.solvers.classical_rk4 import ClassicalRK4Solver
from src.networks.coarse_propagator import CoarsePropagatorNet
from src.networks.trust_gate import TrustGate
from src.solvers.parareal import PararealSolver
from src.training.train_coarse import CoarseTrainer
from src.training.train_k_factor import KFactorTrainer
from src.visualization.plots import (
    plot_trajectories,
    plot_convergence,
    plot_phase_portrait,
    plot_training_loss,
    plot_trust_gate_summary,
    apply_dark_style,
    COLORS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Neural-Accelerated RK4 Solver",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium dark theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }

    .main-header h1 {
        color: white;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 0;
    }

    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-card h3 {
        color: #667eea;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }

    .metric-card .value {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-success {
        background: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }

    .status-warning {
        background: rgba(255, 152, 0, 0.2);
        color: #FF9800;
        border: 1px solid rgba(255, 152, 0, 0.3);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🧮 Neural-Accelerated Parallel RK4 Solver</h1>
    <p>GPU-native hybrid solver combining Parareal decomposition, neural coarse propagation, and adaptive trust gating</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    system_name = st.selectbox(
        "ODE System",
        list(SYSTEM_REGISTRY.keys()),
        format_func=lambda x: get_system(x).name,
        help="Select the benchmark ODE system to solve.",
    )

    system = get_system(system_name)
    params = system.default_params()

    st.markdown("---")
    st.markdown("### 🔧 System Parameters")

    # Dynamic parameter sliders based on system
    edited_params = {}
    ranges = system.param_ranges()
    for pname in system.param_names:
        low, high = ranges[pname]
        default = params[pname]
        edited_params[pname] = st.slider(
            pname, float(low), float(high), float(default),
            step=float((high - low) / 100),
            key=f"param_{pname}",
        )

    st.markdown("---")
    st.markdown("### 📐 Solver Settings")

    dt = st.select_slider(
        "Step size (dt)",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
        value=0.01,
    )

    t_start, t_end = system.default_time_span()
    t_end_slider = st.slider(
        "Integration end time",
        float(t_start), float(t_end * 1.5), float(t_end),
    )

    st.markdown("---")
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"**Device**: {device_name}")
    st.markdown(f"**PyTorch**: {torch.__version__}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Correctness", "🧠 Neural Solver", "🔄 Convergence", "⚡ Benchmarks"
])


# ---------------------------------------------------------------------------
# Tab 1: Correctness — Classical RK4 vs Analytical
# ---------------------------------------------------------------------------

with tab1:
    st.markdown("## Classical RK4 Solver Validation")
    st.markdown(
        "Validates the classical RK4 implementation against analytical solutions "
        "(where available) and scipy's adaptive RK45 solver."
    )

    solver = ClassicalRK4Solver(device=torch.device("cpu"))

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("▶️ Run Classical RK4", key="run_rk4", type="primary"):
            y0 = system.default_initial_condition()
            t_span = (t_start, t_end_slider)

            with st.spinner("Solving..."):
                start = time.time()
                result = solver.solve_single(
                    f=system.f, y0=y0, t_span=t_span,
                    dt=dt, params=edited_params,
                )
                elapsed = time.time() - start

            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(f"""<div class="metric-card">
                    <h3>Steps</h3>
                    <div class="value">{result.y.shape[0]-1:,}</div>
                </div>""", unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""<div class="metric-card">
                    <h3>Wall Time</h3>
                    <div class="value">{elapsed*1000:.1f}ms</div>
                </div>""", unsafe_allow_html=True)
            with col_m3:
                # Check if analytical solution exists
                analytical = system.analytical_solution(result.t, edited_params)
                if analytical is not None:
                    max_err = torch.max(torch.abs(result.y - analytical)).item()
                    st.markdown(f"""<div class="metric-card">
                        <h3>Max Error</h3>
                        <div class="value">{max_err:.2e}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="metric-card">
                        <h3>Max Error</h3>
                        <div class="value">N/A</div>
                    </div>""", unsafe_allow_html=True)

            # Trajectory plot
            names = {
                "damped_oscillator": ["Position x", "Velocity v"],
                "lotka_volterra": ["Prey", "Predator"],
                "van_der_pol": ["Position x", "Velocity v"],
                "lorenz": ["x", "y", "z"],
            }

            fig = plot_trajectories(
                result.t, result.y,
                t_analytical=result.t if analytical is not None else None,
                y_analytical=analytical,
                component_names=names.get(system_name),
                title=f"{system.name} — RK4 Solution (dt={dt})",
            )
            st.pyplot(fig)
            plt.close(fig)

            # Phase portrait for 2-D systems
            if system.dim == 2:
                fig2 = plot_phase_portrait(
                    result.y, y_ref=analytical,
                    xlabel=names.get(system_name, ["x", "v"])[0],
                    ylabel=names.get(system_name, ["x", "v"])[1],
                    title=f"{system.name} — Phase Portrait",
                )
                st.pyplot(fig2)
                plt.close(fig2)


# ---------------------------------------------------------------------------
# Tab 2: Neural Solver — Train + Parareal
# ---------------------------------------------------------------------------

with tab2:
    st.markdown("## Neural-Augmented Parareal Solver")
    st.markdown(
        "Train a neural coarse propagator and run the Parareal algorithm "
        "with adaptive trust gating."
    )

    col_train1, col_train2 = st.columns(2)

    with col_train1:
        st.markdown("### Training Configuration")
        n_traj = st.slider("Training trajectories", 10, 500, 50,
                           step=10, key="n_traj")
        coarse_epochs = st.slider("Training epochs", 100, 10000, 1000,
                                  step=100, key="coarse_epochs")
        coarse_dt_train = st.select_slider(
            "Coarse step size",
            options=[0.05, 0.1, 0.2, 0.5],
            value=0.1, key="coarse_dt",
        )

    with col_train2:
        st.markdown("### Parareal Configuration")
        n_slabs = st.slider("Number of time slabs (P)", 2, 32, 8,
                            key="n_slabs")
        tolerance = st.select_slider(
            "Convergence tolerance",
            options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-8],
            value=1e-4, key="tolerance",
        )
        use_gate = st.checkbox("Enable Trust Gate", value=True, key="use_gate")

    if st.button("🧠 Train & Solve", key="train_solve", type="primary"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Step 1: Train coarse propagator
        st.markdown("### Step 1: Training Neural Coarse Propagator")
        progress_bar = st.progress(0, text="Generating training data...")

        trainer = CoarseTrainer(system, device=device)

        with st.spinner("Training in progress..."):
            model, history = trainer.train(
                n_trajectories=n_traj,
                coarse_dt=coarse_dt_train,
                epochs=coarse_epochs,
                hidden_dim=64,  # smaller for demo speed
            )

        progress_bar.progress(50, text="Training complete!")

        # Training loss plot
        fig_loss = plot_training_loss(
            history, title="Coarse Propagator Training Loss",
        )
        st.pyplot(fig_loss)
        plt.close(fig_loss)

        # Step 2: Parareal solve
        st.markdown("### Step 2: Running Parareal Solver")
        progress_bar.progress(60, text="Running Parareal...")

        y0 = system.default_initial_condition().to(device)
        theta = system.param_vector(edited_params, device=device)

        parareal_solver = PararealSolver(
            coarse_net=model,
            device=device,
            trust_gate=TrustGate(initial_threshold=0.1, decay_rate=0.7),
            max_iterations=30,
            system_name=system_name,
            n_workers=0,
        )

        para_result = parareal_solver.solve(
            f=system.f,
            y0=y0,
            t_span=(t_start, t_end_slider),
            n_slabs=n_slabs,
            fine_dt=dt,
            params=edited_params,
            theta_ode=theta,
            tolerance=tolerance,
            use_trust_gate=use_gate,
        )

        progress_bar.progress(100, text="Done!")

        # Results
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown(f"""<div class="metric-card">
                <h3>Iterations</h3>
                <div class="value">{para_result.n_iterations}</div>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""<div class="metric-card">
                <h3>Wall Time</h3>
                <div class="value">{para_result.wall_time:.2f}s</div>
            </div>""", unsafe_allow_html=True)
        with col_r3:
            final_err = para_result.convergence_history[-1] if para_result.convergence_history else 0
            st.markdown(f"""<div class="metric-card">
                <h3>Final Error</h3>
                <div class="value">{final_err:.2e}</div>
            </div>""", unsafe_allow_html=True)
        with col_r4:
            total_fine = sum(para_result.fine_solves_per_iter)
            st.markdown(f"""<div class="metric-card">
                <h3>Total Fine Solves</h3>
                <div class="value">{total_fine}</div>
            </div>""", unsafe_allow_html=True)

        # Compare with serial RK4
        serial_solver = ClassicalRK4Solver(device=device)
        serial_result = serial_solver.solve_single(
            f=system.f, y0=y0, t_span=(t_start, t_end_slider),
            dt=dt, params=edited_params,
        )

        # Trajectory comparison
        names = {
            "damped_oscillator": ["Position x", "Velocity v"],
            "lotka_volterra": ["Prey", "Predator"],
            "van_der_pol": ["Position x", "Velocity v"],
            "lorenz": ["x", "y", "z"],
        }

        fig_traj = plot_trajectories(
            serial_result.t.cpu(), serial_result.y.cpu(),
            t_parareal=para_result.t.cpu(),
            y_parareal=para_result.y.cpu(),
            component_names=names.get(system_name),
            title=f"{system.name} — RK4 vs Parareal (P={n_slabs})",
        )
        st.pyplot(fig_traj)
        plt.close(fig_traj)


# ---------------------------------------------------------------------------
# Tab 3: Convergence Analysis
# ---------------------------------------------------------------------------

with tab3:
    st.markdown("## Parareal Convergence Analysis")
    st.markdown(
        "Visualise how the Parareal algorithm converges over iterations. "
        "Run the Neural Solver tab first to generate results."
    )

    if st.button("🔄 Run Convergence Analysis", key="run_conv", type="primary"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Quick train + solve for convergence data
        with st.spinner("Training coarse propagator (lightweight)..."):
            trainer = CoarseTrainer(system, device=device)
            model, _ = trainer.train(
                n_trajectories=30, epochs=500, hidden_dim=32,
            )

        y0 = system.default_initial_condition().to(device)
        theta = system.param_vector(edited_params, device=device)
        n_slabs_conv = 8

        with st.spinner("Running Parareal..."):
            parareal_solver = PararealSolver(
                coarse_net=model, device=device,
                trust_gate=TrustGate(initial_threshold=0.1, decay_rate=0.7),
                system_name=system_name,
                n_workers=os.cpu_count() or 4,
            )
            result = parareal_solver.solve(
                f=system.f, y0=y0,
                t_span=(t_start, t_end_slider),
                n_slabs=n_slabs_conv, fine_dt=dt,
                params=edited_params, theta_ode=theta,
                tolerance=1e-6, use_trust_gate=True,
            )

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig_conv = plot_convergence(
                result.convergence_history,
                fine_solves_per_iter=result.fine_solves_per_iter,
                tolerance=1e-6,
                title="Parareal Convergence History",
            )
            st.pyplot(fig_conv)
            plt.close(fig_conv)

        with col_c2:
            if len(result.trust_gate_stats) > 1:
                fig_gate = plot_trust_gate_summary(
                    result.trust_gate_stats,
                    title="Trust Gate Behaviour",
                )
                st.pyplot(fig_gate)
                plt.close(fig_gate)
            else:
                st.info("Trust gate stats require multiple iterations.")

        # Convergence table
        st.markdown("### Iteration Details")
        import pandas as pd
        iter_data = {
            "Iteration": list(range(1, result.n_iterations + 1)),
            "Max |ΔU|": [f"{e:.2e}" for e in result.convergence_history],
            "Fine Solves": result.fine_solves_per_iter,
        }
        if result.trust_gate_stats:
            iter_data["Trust Rate"] = [
                f"{s.get('trust_rate', 0)*100:.0f}%"
                for s in result.trust_gate_stats
            ]
        st.dataframe(pd.DataFrame(iter_data), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Benchmarks
# ---------------------------------------------------------------------------

with tab4:
    st.markdown("## Performance Benchmarks")
    st.markdown(
        "Compare wall-clock times and accuracy across solver configurations."
    )

    if st.button("⚡ Run Benchmark Suite", key="run_bench", type="primary"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y0 = system.default_initial_condition()
        t_span = (t_start, t_end_slider)

        apply_dark_style()

        # Benchmark different dt values
        st.markdown("### Step Size vs Accuracy")
        dt_values = [0.05, 0.01, 0.005, 0.001, 0.0005]
        times_list = []
        errors_list = []
        steps_list = []

        solver = ClassicalRK4Solver(device=torch.device("cpu"))

        # Get reference solution (very fine dt)
        ref_result = solver.solve_single(
            f=system.f, y0=y0, t_span=t_span,
            dt=0.0001, params=edited_params,
        )

        for test_dt in dt_values:
            start = time.time()
            result = solver.solve_single(
                f=system.f, y0=y0, t_span=t_span,
                dt=test_dt, params=edited_params,
            )
            elapsed = time.time() - start
            times_list.append(elapsed * 1000)
            steps_list.append(result.y.shape[0] - 1)

            # Compute error at final point vs reference
            ref_idx = min(result.y.shape[0] - 1, ref_result.y.shape[0] - 1)
            err = torch.max(torch.abs(result.y[-1] - ref_result.y[ref_idx])).item()
            errors_list.append(err)

        # Plot benchmarks
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{system.name} — Step Size Benchmark",
                     fontsize=16, fontweight="bold")

        ax1.loglog(dt_values, errors_list, "o-",
                   color=COLORS["parareal"], linewidth=2, markersize=8)
        ax1.set_xlabel("Step size (dt)", fontsize=12)
        ax1.set_ylabel("Final point error", fontsize=12)
        ax1.set_title("Accuracy vs Step Size", fontsize=13)
        ax1.grid(True, alpha=0.3, which="both")

        ax2.bar(range(len(dt_values)), times_list,
                color=COLORS["rk4"], alpha=0.8)
        ax2.set_xticks(range(len(dt_values)))
        ax2.set_xticklabels([f"{d}" for d in dt_values], rotation=45)
        ax2.set_xlabel("Step size (dt)", fontsize=12)
        ax2.set_ylabel("Wall time (ms)", fontsize=12)
        ax2.set_title("Computation Time", fontsize=13)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Summary table
        import pandas as pd
        bench_df = pd.DataFrame({
            "Step Size": dt_values,
            "Steps": steps_list,
            "Time (ms)": [f"{t:.1f}" for t in times_list],
            "Error": [f"{e:.2e}" for e in errors_list],
        })
        st.dataframe(bench_df, use_container_width=True)

        # System comparison
        st.markdown("### Cross-System Comparison")
        system_times = {}
        test_dt = 0.01

        for sname in SYSTEM_REGISTRY:
            sys_inst = get_system(sname)
            y0_sys = sys_inst.default_initial_condition()
            ts, te = sys_inst.default_time_span()

            start = time.time()
            solver.solve_single(
                f=sys_inst.f, y0=y0_sys,
                t_span=(ts, te), dt=test_dt,
                params=sys_inst.default_params(),
            )
            system_times[sys_inst.name] = (time.time() - start) * 1000

        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
        fig3.suptitle("RK4 Solve Time by System (dt=0.01)",
                      fontsize=16, fontweight="bold")

        names_list = list(system_times.keys())
        vals = list(system_times.values())
        bars = ax3.barh(names_list, vals, color=[
            COLORS["rk4"], COLORS["parareal"],
            COLORS["neural"], COLORS["analytical"]
        ], alpha=0.85)

        for bar, val in zip(bars, vals):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}ms", va="center", fontsize=11, color="#e0e0e0")

        ax3.set_xlabel("Wall time (ms)", fontsize=12)
        ax3.grid(True, alpha=0.3, axis="x")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: rgba(255,255,255,0.4); "
    "font-size: 0.85rem;'>"
    "Neural-Accelerated Parallel RK4 Solver &nbsp;|&nbsp; "
    "Built with PyTorch + Streamlit &nbsp;|&nbsp; "
    f"Device: {device_name}"
    "</div>",
    unsafe_allow_html=True,
)
