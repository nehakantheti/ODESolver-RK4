"""Plotting and visualization utilities for solver comparison.

Provides static and interactive plotting functions for:
    1. Trajectory comparison (RK4 vs Parareal vs analytical)
    2. Parareal convergence animation
    3. k-factor residual accuracy plots
    4. Trust gate statistics dashboard
    5. Benchmark performance charts

All plots support both Matplotlib (static, publication-quality) and
Plotly (interactive, for Streamlit embedding).

Usage:
    >>> from src.visualization.plots import (
    ...     plot_trajectories, plot_convergence, plot_phase_portrait
    ... )
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Use non-interactive backend when imported as module
matplotlib.use("Agg")

# Consistent styling
COLORS = {
    "rk4": "#2196F3",        # blue
    "parareal": "#FF5722",   # deep orange
    "analytical": "#4CAF50", # green
    "neural": "#9C27B0",     # purple
    "k_factor": "#FF9800",   # orange
    "trust_high": "#4CAF50", # green (trusted)
    "trust_low": "#F44336",  # red (corrected)
}

STYLE_CONFIG = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
}


def apply_dark_style():
    """Apply consistent dark-mode styling to matplotlib plots.

    Uses a modern dark colour palette with high-contrast text for
    readability.  Called automatically by all plotting functions.
    """
    plt.rcParams.update(STYLE_CONFIG)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11


def plot_trajectories(
    t_rk4: Tensor,
    y_rk4: Tensor,
    t_parareal: Optional[Tensor] = None,
    y_parareal: Optional[Tensor] = None,
    t_analytical: Optional[Tensor] = None,
    y_analytical: Optional[Tensor] = None,
    component_names: Optional[List[str]] = None,
    title: str = "Trajectory Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot trajectory comparison between solvers.

    Creates a subplot per state component showing the trajectory from
    each available solver overlaid for visual comparison.

    Args:
        t_rk4: Time points for RK4 solution, shape ``(n_points,)``.
        y_rk4: RK4 trajectory, shape ``(n_points, dim)``.
        t_parareal: Time points for Parareal solution (optional).
        y_parareal: Parareal trajectory (optional).
        t_analytical: Time points for analytical solution (optional).
        y_analytical: Analytical trajectory (optional).
        component_names: Names for each state component (e.g., ["x", "v"]).
        title: Plot title.
        save_path: If provided, saves the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    apply_dark_style()

    # Convert tensors to numpy
    t_np = t_rk4.detach().cpu().numpy()
    y_np = y_rk4.detach().cpu().numpy()
    dim = y_np.shape[1]

    if component_names is None:
        component_names = [f"Component {i}" for i in range(dim)]

    fig, axes = plt.subplots(dim, 1, figsize=(12, 4 * dim), sharex=True)
    if dim == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    for i, ax in enumerate(axes):
        # RK4 (always present)
        ax.plot(t_np, y_np[:, i], color=COLORS["rk4"],
                linewidth=2, label="Classical RK4", alpha=0.9)

        # Parareal (if provided)
        if t_parareal is not None and y_parareal is not None:
            tp = t_parareal.detach().cpu().numpy()
            yp = y_parareal.detach().cpu().numpy()
            ax.plot(tp, yp[:, i], "--", color=COLORS["parareal"],
                    linewidth=2, label="Parareal", alpha=0.9)
            ax.scatter(tp, yp[:, i], color=COLORS["parareal"],
                       s=40, zorder=5, edgecolors="white", linewidth=0.5)

        # Analytical (if provided)
        if t_analytical is not None and y_analytical is not None:
            ta = t_analytical.detach().cpu().numpy()
            ya = y_analytical.detach().cpu().numpy()
            ax.plot(ta, ya[:, i], ":", color=COLORS["analytical"],
                    linewidth=2, label="Analytical", alpha=0.7)

        ax.set_ylabel(component_names[i], fontsize=12)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (t)", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Trajectory plot saved to %s", save_path)

    return fig


def plot_convergence(
    convergence_history: List[float],
    fine_solves_per_iter: Optional[List[int]] = None,
    tolerance: float = 1e-6,
    title: str = "Parareal Convergence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Parareal convergence history.

    Shows max error per iteration on a log scale, with the tolerance
    line for reference.  Optionally shows fine solve count per iteration
    on a secondary axis.

    Args:
        convergence_history: Max change per Parareal iteration.
        fine_solves_per_iter: Number of fine solves per iteration.
        tolerance: Convergence tolerance (shown as horizontal line).
        title: Plot title.
        save_path: If provided, saves the figure.

    Returns:
        The matplotlib Figure object.
    """
    apply_dark_style()

    n_iters = len(convergence_history)
    iterations = list(range(1, n_iters + 1))

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Convergence curve
    ax1.semilogy(iterations, convergence_history,
                 "o-", color=COLORS["parareal"], linewidth=2,
                 markersize=8, label="Max error", zorder=3)
    ax1.axhline(y=tolerance, color=COLORS["analytical"],
                linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Tolerance ({tolerance:.0e})")

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Max |ΔU|", fontsize=12, color=COLORS["parareal"])
    ax1.tick_params(axis="y", labelcolor=COLORS["parareal"])
    ax1.grid(True, alpha=0.3, which="both")

    # Fine solve count on secondary axis
    if fine_solves_per_iter is not None:
        ax2 = ax1.twinx()
        ax2.bar(iterations, fine_solves_per_iter,
                alpha=0.3, color=COLORS["rk4"], label="Fine solves")
        ax2.set_ylabel("Fine solves", fontsize=12, color=COLORS["rk4"])
        ax2.tick_params(axis="y", labelcolor=COLORS["rk4"])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if fine_solves_per_iter is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper right", fontsize=10, framealpha=0.7)
    else:
        ax1.legend(loc="upper right", fontsize=10, framealpha=0.7)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Convergence plot saved to %s", save_path)

    return fig


def plot_phase_portrait(
    y: Tensor,
    y_ref: Optional[Tensor] = None,
    xlabel: str = "x",
    ylabel: str = "v",
    title: str = "Phase Portrait",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a 2-D phase portrait (state space trajectory).

    Shows the trajectory in the (component_0, component_1) plane,
    with optional reference overlay.

    Args:
        y: Trajectory, shape ``(n_points, 2+)``.
        y_ref: Optional reference trajectory for overlay.
        xlabel: Label for x-axis (first component).
        ylabel: Label for y-axis (second component).
        title: Plot title.
        save_path: If provided, saves the figure.

    Returns:
        The matplotlib Figure object.
    """
    apply_dark_style()

    y_np = y.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    if y_ref is not None:
        yr = y_ref.detach().cpu().numpy()
        ax.plot(yr[:, 0], yr[:, 1], color=COLORS["analytical"],
                linewidth=1.5, alpha=0.5, label="Reference")

    ax.plot(y_np[:, 0], y_np[:, 1], color=COLORS["rk4"],
            linewidth=2, label="Solver")
    ax.scatter(y_np[0, 0], y_np[0, 1], color=COLORS["parareal"],
               s=100, zorder=5, marker="o", edgecolors="white",
               linewidth=1, label="IC")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


def plot_training_loss(
    history: Dict[str, List[float]],
    title: str = "Training Loss",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        history: Dictionary with ``"train_loss"`` and ``"val_loss"`` keys.
        title: Plot title.
        save_path: If provided, saves the figure.

    Returns:
        The matplotlib Figure object.
    """
    apply_dark_style()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.semilogy(epochs, history["train_loss"],
                color=COLORS["rk4"], linewidth=2, label="Train loss")
    ax.semilogy(epochs, history["val_loss"],
                color=COLORS["parareal"], linewidth=2,
                linestyle="--", label="Validation loss")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (log scale)", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.7)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


def plot_trust_gate_summary(
    trust_gate_stats: List[dict],
    title: str = "Trust Gate Behaviour",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the trust gate decisions across Parareal iterations.

    Shows the trust rate (fraction of slabs skipped) and threshold
    decay over iterations.

    Args:
        trust_gate_stats: Per-iteration stats from ``PararealResult``.
        title: Plot title.
        save_path: If provided, saves the figure.

    Returns:
        The matplotlib Figure object.
    """
    apply_dark_style()

    n_iters = len(trust_gate_stats)
    iterations = list(range(1, n_iters + 1))

    trust_rates = [s.get("trust_rate", 0) * 100 for s in trust_gate_stats]
    thresholds = [s.get("threshold", 0) for s in trust_gate_stats]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax1.bar(iterations, trust_rates, color=COLORS["trust_high"],
            alpha=0.7, label="Trust rate (%)")
    ax1.set_ylabel("Trust Rate (%)", fontsize=12, color=COLORS["trust_high"])
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLORS["trust_high"])

    ax2 = ax1.twinx()
    ax2.plot(iterations, thresholds, "o-", color=COLORS["neural"],
             linewidth=2, markersize=6, label="Threshold τ")
    ax2.set_ylabel("Threshold τ", fontsize=12, color=COLORS["neural"])
    ax2.tick_params(axis="y", labelcolor=COLORS["neural"])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=10, framealpha=0.7)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig
