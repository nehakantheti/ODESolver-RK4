"""Benchmark suite for comparing solver performance.

Runs systematic benchmarks comparing:
    1. Classical RK4 at various step sizes
    2. Parareal with neural coarse propagator
    3. Serial vs batched execution modes

Results are printed to console and saved to a CSV file.

Usage:
    cd final
    py -3.11 benchmarks/benchmark_solvers.py
"""

from __future__ import annotations

import logging
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd

from src.ode_systems import get_system, SYSTEM_REGISTRY
from src.solvers.classical_rk4 import ClassicalRK4Solver
from src.networks.coarse_propagator import CoarsePropagatorNet
from src.networks.trust_gate import TrustGate
from src.solvers.parareal import PararealSolver
from src.training.train_coarse import CoarseTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def benchmark_step_sizes():
    """Benchmark classical RK4 across step sizes for all ODE systems.

    Measures wall-clock time and accuracy (vs fine reference solution)
    for step sizes from 0.05 down to 0.0001.

    Returns:
        DataFrame with columns: System, dt, steps, time_ms, error.
    """
    logger.info("=" * 60)
    logger.info("BENCHMARK: Step Size Sweep")
    logger.info("=" * 60)

    solver = ClassicalRK4Solver(device=torch.device("cpu"))
    dt_values = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    results = []

    for system_name in SYSTEM_REGISTRY:
        system = get_system(system_name)
        y0 = system.default_initial_condition()
        t_span = system.default_time_span()
        params = system.default_params()

        # Reference solution (very fine)
        ref = solver.solve_single(
            f=system.f, y0=y0, t_span=t_span,
            dt=0.00005, params=params,
        )

        for dt_val in dt_values:
            # Warm up
            solver.solve_single(f=system.f, y0=y0, t_span=t_span,
                                dt=dt_val, params=params)

            # Timed run (average of 3)
            times = []
            for _ in range(3):
                start = time.perf_counter()
                result = solver.solve_single(
                    f=system.f, y0=y0, t_span=t_span,
                    dt=dt_val, params=params,
                )
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times) * 1000  # ms

            # Error at final point
            err = torch.max(torch.abs(result.y[-1] - ref.y[-1])).item()

            results.append({
                "System": system.name,
                "dt": dt_val,
                "steps": result.y.shape[0] - 1,
                "time_ms": round(avg_time, 2),
                "error": err,
            })

            logger.info(
                "%s | dt=%.5f | steps=%d | time=%.2fms | error=%.2e",
                system.name, dt_val, result.y.shape[0] - 1, avg_time, err,
            )

    return pd.DataFrame(results)


def benchmark_parareal_slabs():
    """Benchmark Parareal solver with varying slab counts.

    Trains a coarse propagator (lightweight) and measures wall time
    for different numbers of slabs on the damped oscillator.

    Returns:
        DataFrame with columns: n_slabs, iterations, time_ms, error.
    """
    logger.info("=" * 60)
    logger.info("BENCHMARK: Parareal Slab Count")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = get_system("damped_oscillator")

    # Quick-train a coarse propagator
    logger.info("Training coarse propagator for benchmark...")
    trainer = CoarseTrainer(system, device=device)
    model, _ = trainer.train(
        n_trajectories=50, epochs=500, hidden_dim=32,
    )

    y0 = system.default_initial_condition().to(device)
    params = system.default_params()
    theta = system.param_vector(params, device=device)

    slab_counts = [2, 4, 8, 12, 16]
    results = []

    # Serial reference time
    solver = ClassicalRK4Solver(device=device)
    start = time.perf_counter()
    serial_result = solver.solve_single(
        f=system.f, y0=y0.cpu(), t_span=system.default_time_span(),
        dt=0.01, params=params,
    )
    serial_time = (time.perf_counter() - start) * 1000
    logger.info("Serial RK4 time: %.2fms", serial_time)

    for n_slabs in slab_counts:
        parareal = PararealSolver(
            coarse_net=model, device=device,
            trust_gate=TrustGate(initial_threshold=0.1),
            max_iterations=30,
        )

        start = time.perf_counter()
        result = parareal.solve(
            f=system.f, y0=y0, t_span=system.default_time_span(),
            n_slabs=n_slabs, fine_dt=0.01,
            params=params, theta_ode=theta,
            tolerance=1e-5, use_trust_gate=True,
        )
        elapsed = (time.perf_counter() - start) * 1000

        # Error vs serial at matched endpoints
        endpoint_err = torch.max(
            torch.abs(result.y[-1].cpu() - serial_result.y[-1])
        ).item()

        results.append({
            "n_slabs": n_slabs,
            "iterations": result.n_iterations,
            "time_ms": round(elapsed, 2),
            "serial_time_ms": round(serial_time, 2),
            "speedup": round(serial_time / elapsed, 2) if elapsed > 0 else 0,
            "endpoint_error": endpoint_err,
        })

        logger.info(
            "P=%d | iters=%d | time=%.2fms | speedup=%.2fx | err=%.2e",
            n_slabs, result.n_iterations, elapsed,
            serial_time / elapsed if elapsed > 0 else 0, endpoint_err,
        )

    return pd.DataFrame(results)


def main():
    """Run all benchmarks and save results."""
    logger.info("Starting benchmark suite...")
    logger.info("Device: %s", "CUDA" if torch.cuda.is_available() else "CPU")
    logger.info("PyTorch: %s", torch.__version__)

    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark 1: Step sizes
    df_steps = benchmark_step_sizes()
    df_steps.to_csv(output_dir / "step_size_benchmark.csv", index=False)
    logger.info("Step size results saved to %s",
                output_dir / "step_size_benchmark.csv")
    print("\n" + df_steps.to_string(index=False))

    # Benchmark 2: Parareal slabs
    print("\n")
    df_slabs = benchmark_parareal_slabs()
    df_slabs.to_csv(output_dir / "parareal_benchmark.csv", index=False)
    logger.info("Parareal results saved to %s",
                output_dir / "parareal_benchmark.csv")
    print("\n" + df_slabs.to_string(index=False))

    logger.info("All benchmarks complete!")


if __name__ == "__main__":
    main()
