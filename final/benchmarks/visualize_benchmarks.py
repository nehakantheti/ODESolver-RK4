"""Comprehensive benchmark visualization for the Neural Parareal solver.

Trains models, runs benchmarks, and generates publication-quality
comparison charts inspired by RandNet-Parareal's analysis approach.

Charts generated:
    1. CPU vs GPU: Serial RK4 runtime comparison
    2. Parareal speedup vs slab count
    3. Parareal convergence (max_change per iteration)
    4. Training convergence: Adam → L-BFGS transition
    5. Error vs speedup tradeoff

Usage:
    py -3.11 benchmarks/visualize_benchmarks.py

Output:
    benchmarks/figures/*.png
"""

from __future__ import annotations

import logging
import os
import sys
import time

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.ode_systems import get_system
from src.solvers.classical_rk4 import ClassicalRK4Solver
from src.solvers.parareal import PararealSolver
from src.networks.trust_gate import TrustGate
from src.training.train_coarse import CoarseTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Professional color palette
C = {
    "cpu":       "#3498db",
    "gpu":       "#e74c3c",
    "parareal":  "#2ecc71",
    "adam":       "#f39c12",
    "lbfgs":     "#9b59b6",
    "accent":    "#1abc9c",
    "grid":      "#444444",
    "bg":        "#1a1a2e",
    "text":      "#e0e0e0",
}

# Plot style
plt.rcParams.update({
    "figure.facecolor":   C["bg"],
    "axes.facecolor":     "#16213e",
    "axes.edgecolor":     C["grid"],
    "axes.labelcolor":    C["text"],
    "axes.titlesize":     14,
    "axes.labelsize":     12,
    "xtick.color":        C["text"],
    "ytick.color":        C["text"],
    "text.color":         C["text"],
    "legend.facecolor":   "#16213e",
    "legend.edgecolor":   C["grid"],
    "grid.color":         C["grid"],
    "grid.alpha":         0.3,
    "font.family":        "sans-serif",
    "font.size":          11,
    "figure.figsize":     (10, 6),
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  C["bg"],
})


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _timed(fn, device, n_runs=3):
    """Time a function call with device-appropriate precision."""
    if device.type == "cuda":
        fn()  # warmup
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(n_runs):
            start_evt.record()
            result = fn()
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))
        return result, float(np.mean(times))
    else:
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            result = fn()
            times.append((time.perf_counter() - t0) * 1000)
        return result, float(np.mean(times))


# ---------------------------------------------------------------------------
# Chart 1: CPU vs GPU Serial RK4
# ---------------------------------------------------------------------------

def chart_cpu_vs_gpu(system):
    """Bar chart comparing serial RK4 time on CPU vs GPU."""
    logger.info("Chart 1: CPU vs GPU serial comparison...")

    step_counts = [500, 2000, 10000, 50000]
    cpu_times = []
    gpu_times = []

    cpu_solver = ClassicalRK4Solver(device=torch.device("cpu"))
    gpu_solver = ClassicalRK4Solver(device=DEVICE)

    y0_cpu = system.default_initial_condition()
    y0_gpu = y0_cpu.to(DEVICE)
    params = system.default_params()
    t_span = system.default_time_span()

    for n_steps in step_counts:
        dt = (t_span[1] - t_span[0]) / n_steps

        def _cpu():
            return cpu_solver.solve_single(
                f=system.f, y0=y0_cpu, t_span=t_span, dt=dt, params=params,
            )

        def _gpu():
            return gpu_solver.solve_single(
                f=system.f, y0=y0_gpu, t_span=t_span, dt=dt, params=params,
            )

        _, ct = _timed(_cpu, torch.device("cpu"), n_runs=3)
        _, gt = _timed(_gpu, DEVICE, n_runs=3)
        cpu_times.append(ct)
        gpu_times.append(gt)
        logger.info("  steps=%d  CPU=%.1fms  GPU=%.1fms", n_steps, ct, gt)

    # Plot
    fig, ax = plt.subplots()
    x = np.arange(len(step_counts))
    w = 0.35

    bars1 = ax.bar(x - w/2, cpu_times, w, label="CPU (Sequential)",
                   color=C["cpu"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, gpu_times, w, label="GPU (Sequential)",
                   color=C["gpu"], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Number of RK4 Steps")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Serial RK4: CPU vs GPU Runtime")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in step_counts])
    ax.legend()
    ax.grid(True, axis="y")
    ax.set_yscale("log")

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8, color=C["text"])
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8, color=C["text"])

    path = os.path.join(FIG_DIR, "1_cpu_vs_gpu_serial.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved: %s", path)
    return cpu_times, gpu_times, step_counts


# ---------------------------------------------------------------------------
# Chart 2: Parareal Speedup vs Slab Count
# ---------------------------------------------------------------------------

def chart_parareal_speedup(system, model, fine_dts=[0.01, 0.001]):
    """Bar chart of Parareal speedup over serial CPU for different dt values."""
    logger.info("Chart 2: Parareal speedup vs slab count...")

    slab_counts = [4, 8, 16]
    params = system.default_params()
    theta = system.param_vector(params, device=DEVICE)

    y0_cpu = system.default_initial_condition()
    y0_gpu = y0_cpu.to(DEVICE)
    t_span = system.default_time_span()

    cpu_solver = ClassicalRK4Solver(device=torch.device("cpu"))

    all_results = {}
    for fine_dt in fine_dts:
        # Serial CPU baseline
        def _serial():
            return cpu_solver.solve_single(
                f=system.f, y0=y0_cpu, t_span=t_span,
                dt=fine_dt, params=params,
            )
        _, serial_ms = _timed(_serial, torch.device("cpu"), n_runs=3)
        logger.info("  Serial CPU (dt=%.4f): %.1fms", fine_dt, serial_ms)

        speedups = []
        for P in slab_counts:
            parareal = PararealSolver(
                coarse_net=model, device=DEVICE,
                trust_gate=TrustGate(initial_threshold=0.1),
                max_iterations=50,
                system_name="damped_oscillator",
                n_workers=os.cpu_count() or 4,
            )

            def _para():
                return parareal.solve(
                    f=system.f, y0=y0_gpu, t_span=t_span,
                    n_slabs=P, fine_dt=fine_dt,
                    params=params, theta_ode=theta,
                    tolerance=1e-6, use_trust_gate=True,
                )

            result, para_ms = _timed(_para, DEVICE, n_runs=1)
            spd = serial_ms / para_ms if para_ms > 0 else 0
            speedups.append(spd)
            logger.info("    P=%d: %.1fms (%.2fx speedup, K=%d)",
                        P, para_ms, spd, result.n_iterations)

        all_results[fine_dt] = (serial_ms, speedups)

    # Plot
    fig, ax = plt.subplots()
    x = np.arange(len(slab_counts))
    w = 0.35
    colors = [C["parareal"], C["accent"]]

    for i, fine_dt in enumerate(fine_dts):
        serial_ms, speedups = all_results[fine_dt]
        n_steps = int((t_span[1] - t_span[0]) / fine_dt)
        label = f"dt={fine_dt} ({n_steps:,} steps)"
        bars = ax.bar(x + (i - 0.5) * w, speedups, w, label=label,
                      color=colors[i % len(colors)],
                      edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f"{h:.2f}x", ha="center", va="bottom",
                    fontsize=9, color=C["text"], fontweight="bold")

    ax.axhline(y=1.0, color=C["gpu"], linestyle="--", alpha=0.7,
               label="Breakeven (1.0x)")
    ax.set_xlabel("Number of Slabs (P)")
    ax.set_ylabel("Speedup vs Serial CPU")
    ax.set_title("Parareal GPU Speedup over Serial CPU RK4")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in slab_counts])
    ax.legend()
    ax.grid(True, axis="y")

    path = os.path.join(FIG_DIR, "2_parareal_speedup.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved: %s", path)
    return all_results


# ---------------------------------------------------------------------------
# Chart 3: Parareal Convergence
# ---------------------------------------------------------------------------

def chart_convergence(system, model, slab_counts=[4, 8, 16]):
    """Line plot of max_change per Parareal iteration."""
    logger.info("Chart 3: Parareal convergence per iteration...")

    params = system.default_params()
    theta = system.param_vector(params, device=DEVICE)
    y0 = system.default_initial_condition().to(DEVICE)
    t_span = system.default_time_span()

    fig, ax = plt.subplots()
    colors_iter = [C["cpu"], C["parareal"], C["accent"]]

    for i, P in enumerate(slab_counts):
        parareal = PararealSolver(
            coarse_net=model, device=DEVICE,
            trust_gate=TrustGate(initial_threshold=0.1),
            max_iterations=50,
            system_name="damped_oscillator",
            n_workers=os.cpu_count() or 4,
        )
        result = parareal.solve(
            f=system.f, y0=y0, t_span=t_span,
            n_slabs=P, fine_dt=0.001,
            params=params, theta_ode=theta,
            tolerance=1e-6, use_trust_gate=True,
        )

        hist = result.convergence_history
        iters = list(range(len(hist)))
        ax.semilogy(iters, hist, "o-", color=colors_iter[i % 3],
                     label=f"P={P} (K={len(hist)})", linewidth=2,
                     markersize=5)

    ax.axhline(y=1e-6, color=C["gpu"], linestyle=":", alpha=0.7,
               label="Tolerance (1e-6)")
    ax.set_xlabel("Iteration (k)")
    ax.set_ylabel("Max Change (log scale)")
    ax.set_title("Parareal Convergence: Multi-Step Coarse Propagation")
    ax.legend()
    ax.grid(True)

    path = os.path.join(FIG_DIR, "3_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Chart 4: Training Convergence (Adam → L-BFGS)
# ---------------------------------------------------------------------------

def chart_training_convergence(history_hybrid, adam_epochs):
    """Loss curve showing Adam phase and L-BFGS phase side by side."""
    logger.info("Chart 4: Training convergence (Adam → L-BFGS)...")

    train_loss = history_hybrid["train_loss"]
    val_loss = history_hybrid["val_loss"]
    total = len(train_loss)

    fig, ax = plt.subplots()

    # Adam phase
    adam_train = train_loss[:adam_epochs]
    adam_val = val_loss[:adam_epochs]
    ax.semilogy(range(len(adam_train)), adam_train, color=C["adam"],
                alpha=0.8, linewidth=1.5, label="Adam: train")
    ax.semilogy(range(len(adam_val)), adam_val, color=C["adam"],
                alpha=0.4, linewidth=1, linestyle="--", label="Adam: val")

    # L-BFGS phase
    if total > adam_epochs:
        lbfgs_train = train_loss[adam_epochs:]
        lbfgs_val = val_loss[adam_epochs:]
        lbfgs_x = range(adam_epochs, total)
        ax.semilogy(lbfgs_x, lbfgs_train, color=C["lbfgs"],
                     linewidth=2, label="L-BFGS: train")
        ax.semilogy(lbfgs_x, lbfgs_val, color=C["lbfgs"],
                     alpha=0.5, linewidth=1, linestyle="--",
                     label="L-BFGS: val")

        # Transition line
        ax.axvline(x=adam_epochs, color=C["text"], linestyle=":",
                   alpha=0.5, label="Adam → L-BFGS")

        # Annotate improvement
        adam_final = adam_train[-1] if adam_train else 0
        lbfgs_final = lbfgs_train[-1] if lbfgs_train else 0
        if adam_final > 0 and lbfgs_final > 0:
            improvement = adam_final / lbfgs_final
            ax.annotate(
                f"{improvement:.1f}× lower loss",
                xy=(total - 1, lbfgs_final),
                xytext=(total * 0.7, adam_final * 0.5),
                arrowprops=dict(arrowstyle="->", color=C["accent"]),
                color=C["accent"], fontsize=10, fontweight="bold",
            )

    ax.set_xlabel("Epoch / Step")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Hybrid Training: Adam → L-BFGS Convergence")
    ax.legend(loc="upper right")
    ax.grid(True)

    path = os.path.join(FIG_DIR, "4_training_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Chart 5: Error vs Speedup Tradeoff
# ---------------------------------------------------------------------------

def chart_error_vs_speedup(system, model):
    """Scatter plot of endpoint error vs speedup for different configurations."""
    logger.info("Chart 5: Error vs speedup tradeoff...")

    params = system.default_params()
    theta = system.param_vector(params, device=DEVICE)
    y0_cpu = system.default_initial_condition()
    y0_gpu = y0_cpu.to(DEVICE)
    t_span = system.default_time_span()

    cpu_solver = ClassicalRK4Solver(device=torch.device("cpu"))

    configs = [
        (0.01, [4, 8, 16]),
        (0.001, [4, 8, 16]),
    ]

    fig, ax = plt.subplots()
    markers = ["o", "s"]
    colors = [C["parareal"], C["accent"]]

    for ci, (fine_dt, slabs) in enumerate(configs):
        # Serial baseline
        def _serial():
            return cpu_solver.solve_single(
                f=system.f, y0=y0_cpu, t_span=t_span,
                dt=fine_dt, params=params,
            )
        serial_result, serial_ms = _timed(_serial, torch.device("cpu"), n_runs=3)

        for P in slabs:
            parareal = PararealSolver(
                coarse_net=model, device=DEVICE,
                trust_gate=TrustGate(initial_threshold=0.1),
                max_iterations=50,
                system_name="damped_oscillator",
                n_workers=os.cpu_count() or 4,
            )

            def _para():
                return parareal.solve(
                    f=system.f, y0=y0_gpu, t_span=t_span,
                    n_slabs=P, fine_dt=fine_dt,
                    params=params, theta_ode=theta,
                    tolerance=1e-6, use_trust_gate=True,
                )

            result, para_ms = _timed(_para, DEVICE, n_runs=1)

            err = torch.max(
                torch.abs(result.y[-1].cpu() - serial_result.y[-1].cpu())
            ).item()
            spd = serial_ms / para_ms if para_ms > 0 else 0

            n_steps = int((t_span[1] - t_span[0]) / fine_dt)
            ax.scatter(spd, err, s=100, marker=markers[ci],
                       color=colors[ci], edgecolors="white", linewidth=0.5,
                       zorder=5)
            ax.annotate(f"P={P}", (spd, err), textcoords="offset points",
                        xytext=(8, 4), fontsize=9, color=C["text"])

    # Add legend entries manually
    for ci, (fine_dt, _) in enumerate(configs):
        n_steps = int((t_span[1] - t_span[0]) / fine_dt)
        ax.scatter([], [], marker=markers[ci], color=colors[ci],
                   label=f"dt={fine_dt} ({n_steps:,} steps)", s=80)

    ax.axvline(x=1.0, color=C["gpu"], linestyle="--", alpha=0.5,
               label="Breakeven")
    ax.set_xlabel("Speedup vs Serial CPU")
    ax.set_ylabel("Endpoint Error (max abs)")
    ax.set_title("Accuracy vs Speedup Tradeoff")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

    path = os.path.join(FIG_DIR, "5_error_vs_speedup.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run all benchmarks and generate visualizations."""
    logger.info("=" * 60)
    logger.info("NEURAL PARAREAL BENCHMARK VISUALIZATION")
    logger.info("=" * 60)
    logger.info("Device: %s", DEVICE)
    if DEVICE.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(DEVICE))
    logger.info("Output: %s", FIG_DIR)
    logger.info("")

    system = get_system("damped_oscillator")

    # -- Train coarse propagator with hybrid optimizer ----------------------
    logger.info("Training coarse propagator (hybrid Adam → L-BFGS)...")
    adam_epochs = 500
    lbfgs_steps = 50

    trainer = CoarseTrainer(system, device=DEVICE)
    model, history = trainer.train(
        n_trajectories=100,
        epochs=adam_epochs,
        hidden_dim=64,
        lbfgs_steps=lbfgs_steps,
    )
    logger.info("Training complete. Generating charts...\n")

    # -- Generate all charts ------------------------------------------------
    chart_cpu_vs_gpu(system)
    chart_parareal_speedup(system, model)
    chart_convergence(system, model)
    chart_training_convergence(history, adam_epochs)
    chart_error_vs_speedup(system, model)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL CHARTS SAVED TO: %s", FIG_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
