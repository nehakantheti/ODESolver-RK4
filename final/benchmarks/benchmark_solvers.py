"""Benchmark suite for comparing solver performance.

Runs systematic benchmarks comparing:
    1. Classical RK4 at various step sizes (GPU-accelerated)
    2. Parareal with neural coarse propagator
    3. Serial vs parallel execution modes

GPU usage:
    - All solvers run on CUDA if available
    - CUDA events used for accurate GPU timing
    - torch.cuda.synchronize() ensures timing accuracy

Results are printed to console and saved to CSV files.

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

# Auto-detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _timed_call(fn, device, n_runs=3):
    """Time a callable accurately on both CPU and GPU.

    On CUDA: uses torch.cuda.Event for precise GPU timing.
    On CPU: uses time.perf_counter.

    Args:
        fn: Callable to time (no arguments).
        device: Torch device being used.
        n_runs: Number of repetitions to average.

    Returns:
        Tuple of (result_from_last_call, avg_time_ms).
    """
    if device.type == "cuda":
        # Warmup
        result = fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = fn()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        # Warmup
        result = fn()

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = fn()
            times.append((time.perf_counter() - start) * 1000)

    return result, sum(times) / len(times)


def benchmark_step_sizes():
    """Benchmark classical RK4 across step sizes for all ODE systems.

    All solves run on the auto-detected device (GPU if available).
    Measures wall-clock time and accuracy (vs fine reference solution)
    for step sizes from 0.05 down to 0.0001.

    Returns:
        DataFrame with columns: System, dt, steps, time_ms, error.
    """
    logger.info("=" * 60)
    logger.info("BENCHMARK: Step Size Sweep (device=%s)", DEVICE)
    logger.info("=" * 60)

    solver = ClassicalRK4Solver(device=DEVICE)
    dt_values = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    results = []

    for system_name in SYSTEM_REGISTRY:
        system = get_system(system_name)
        y0 = system.default_initial_condition().to(DEVICE)
        t_span = system.default_time_span()
        params = system.default_params()

        # Reference solution (very fine)
        ref = solver.solve_single(
            f=system.f, y0=y0, t_span=t_span,
            dt=0.00005, params=params,
        )

        for dt_val in dt_values:
            def _run():
                return solver.solve_single(
                    f=system.f, y0=y0, t_span=t_span,
                    dt=dt_val, params=params,
                )

            result, avg_time = _timed_call(_run, DEVICE)

            # Error at final point (both on same device)
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

    Everything runs on the auto-detected device.
    Trains a coarse propagator (lightweight) and measures wall time
    for different numbers of slabs on the damped oscillator.

    Returns:
        DataFrame with columns: n_slabs, iterations, time_ms, error.
    """
    logger.info("=" * 60)
    logger.info("BENCHMARK: Parareal Slab Count (device=%s)", DEVICE)
    logger.info("=" * 60)

    system = get_system("damped_oscillator")

    # Quick-train a coarse propagator on GPU
    logger.info("Training coarse propagator for benchmark...")
    trainer = CoarseTrainer(system, device=DEVICE)
    model, _ = trainer.train(
        n_trajectories=50, epochs=500, hidden_dim=32,
    )

    y0_cpu = system.default_initial_condition()
    y0_gpu = y0_cpu.to(DEVICE)
    params = system.default_params()
    theta = system.param_vector(params, device=DEVICE)

    slab_counts = [2, 4, 8, 12, 16]
    results = []

    # Serial reference: CPU is faster for sequential RK4 (no kernel overhead)
    cpu_solver = ClassicalRK4Solver(device=torch.device("cpu"))

    def _serial_run():
        return cpu_solver.solve_single(
            f=system.f, y0=y0_cpu, t_span=system.default_time_span(),
            dt=0.01, params=params,
        )

    serial_result, serial_time = _timed_call(_serial_run, torch.device("cpu"))
    logger.info("Serial RK4 baseline: %.2fms (CPU)", serial_time)

    for n_slabs in slab_counts:
        parareal = PararealSolver(
            coarse_net=model, device=DEVICE,
            trust_gate=TrustGate(initial_threshold=0.1),
            max_iterations=30,
            system_name="damped_oscillator",
            n_workers=0,  # CPU vmap (faster than multiprocessing for typical workloads)
        )

        def _parareal_run():
            return parareal.solve(
                f=system.f, y0=y0_gpu, t_span=system.default_time_span(),
                n_slabs=n_slabs, fine_dt=0.01,
                params=params, theta_ode=theta,
                tolerance=1e-5, use_trust_gate=True,
            )

        result, elapsed = _timed_call(_parareal_run, DEVICE, n_runs=1)

        # Error vs serial: move GPU result to CPU for comparison
        endpoint_err = torch.max(
            torch.abs(result.y[-1].cpu() - serial_result.y[-1].cpu())
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


def benchmark_parareal_hard():
    """Benchmark Parareal on a HARD problem (fine dt=0.001, 20K steps).

    This is the regime where Parareal is designed to shine: the serial
    solve is expensive (20,000 steps → ~3-4 seconds on CPU), so the
    parallel batched fine pass can amortise its overhead.

    Uses a better-trained coarse propagator (more data, larger model,
    more epochs) to reduce the iteration count K.

    Returns:
        DataFrame with columns: n_slabs, iterations, time_ms, error.
    """
    logger.info("=" * 60)
    logger.info("BENCHMARK: Parareal HARD (fine_dt=0.001, device=%s)", DEVICE)
    logger.info("=" * 60)

    system = get_system("damped_oscillator")

    # Better-trained coarse propagator for fewer Parareal iterations
    logger.info("Training stronger coarse propagator (hidden=64, epochs=1000)...")
    trainer = CoarseTrainer(system, device=DEVICE)
    model, _ = trainer.train(
        n_trajectories=100, epochs=1000, hidden_dim=64,
    )

    y0_cpu = system.default_initial_condition()
    y0_gpu = y0_cpu.to(DEVICE)
    params = system.default_params()
    theta = system.param_vector(params, device=DEVICE)

    fine_dt = 0.001  # 10× finer → 10× more serial work

    slab_counts = [2, 4, 8, 16]
    results = []

    # Serial reference on CPU (sequential RK4)
    cpu_solver = ClassicalRK4Solver(device=torch.device("cpu"))

    def _serial_run():
        return cpu_solver.solve_single(
            f=system.f, y0=y0_cpu, t_span=system.default_time_span(),
            dt=fine_dt, params=params,
        )

    serial_result, serial_time = _timed_call(_serial_run, torch.device("cpu"))
    logger.info(
        "Serial RK4 baseline: %.2fms (CPU, dt=%.4f, steps=%d)",
        serial_time, fine_dt,
        int((system.default_time_span()[1] - system.default_time_span()[0]) / fine_dt),
    )

    for n_slabs in slab_counts:
        parareal = PararealSolver(
            coarse_net=model, device=DEVICE,
            trust_gate=TrustGate(initial_threshold=0.1),
            max_iterations=50,
            system_name="damped_oscillator",
            n_workers=0,  # CPU vmap (faster than multiprocessing for typical workloads)
        )

        def _parareal_run():
            return parareal.solve(
                f=system.f, y0=y0_gpu, t_span=system.default_time_span(),
                n_slabs=n_slabs, fine_dt=fine_dt,
                params=params, theta_ode=theta,
                tolerance=1e-6, use_trust_gate=True,
            )

        result, elapsed = _timed_call(_parareal_run, DEVICE, n_runs=1)

        # Error vs serial: move GPU result to CPU for comparison
        endpoint_err = torch.max(
            torch.abs(result.y[-1].cpu() - serial_result.y[-1].cpu())
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
    logger.info("Device: %s", DEVICE)
    if DEVICE.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(DEVICE))
    logger.info("PyTorch: %s", torch.__version__)

    output_dir = Path("benchmarks/results/server-runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark 1: Step sizes
    df_steps = benchmark_step_sizes()
    df_steps.to_csv(output_dir / "step_size_benchmark.csv", index=False)
    logger.info("Step size results saved to %s",
                output_dir / "step_size_benchmark.csv")
    print("\n" + df_steps.to_string(index=False))

    # Benchmark 2: Parareal slabs (easy — dt=0.01, 2K steps)
    print("\n")
    df_easy = benchmark_parareal_slabs()
    df_easy.to_csv(output_dir / "parareal_easy_benchmark.csv", index=False)
    logger.info("Parareal EASY results saved to %s",
                output_dir / "parareal_easy_benchmark.csv")
    print("\n" + df_easy.to_string(index=False))

    # Benchmark 3: Parareal slabs (hard — dt=0.001, 20K steps)
    print("\n")
    df_hard = benchmark_parareal_hard()
    df_hard.to_csv(output_dir / "parareal_hard_benchmark.csv", index=False)
    logger.info("Parareal HARD results saved to %s",
                output_dir / "parareal_hard_benchmark.csv")
    print("\n" + df_hard.to_string(index=False))

    logger.info("All benchmarks complete!")


if __name__ == "__main__":
    main()

