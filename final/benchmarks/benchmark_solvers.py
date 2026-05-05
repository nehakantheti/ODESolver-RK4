"""Benchmark suite for comparing solver performance.

Runs systematic benchmarks comparing:
    1. Classical RK4 at various step sizes
    2. Parareal with neural coarse propagator
    3. Serial vs parallel execution modes

Three benchmark modes:
    - **cpu**: Everything on CPU, sequential fine pass (baseline).
    - **gpu**: Coarse net on CUDA, fine pass on CPU sequential.
    - **multiproc**: Coarse net on CPU, fine pass with multiprocessing workers.

Each mode saves results to ``benchmarks/results/<mode>/`` for comparison.

Usage:
    cd final
    python benchmarks/benchmark_solvers.py --mode cpu
    python benchmarks/benchmark_solvers.py --mode gpu
    python benchmarks/benchmark_solvers.py --mode multiproc
"""

from __future__ import annotations

import argparse
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

# Auto-detect optimal worker count for multiprocessing fine pass
# Quarter of logical cores avoids SMT contention (e.g. 64 threads → 16 workers)
N_WORKERS = min(os.cpu_count() // 4, 16) if os.cpu_count() else 0


def _load_coarse_model(system, device, hidden_dim=128):
    """Load a pre-trained coarse propagator from trained_models/.

    Looks for ``trained_models/coarse_{system_name}.pt``.  Falls back
    to lightweight training if the model file is not found.

    Args:
        system: ODE system instance.
        device: Torch device.
        hidden_dim: Hidden layer width (must match the saved model).

    Returns:
        Loaded CoarsePropagatorNet in eval mode.
    """
    sys_key = system.name.lower().replace(" ", "_")
    model_path = Path("trained_models") / f"coarse_{sys_key}.pt"

    net = CoarsePropagatorNet(
        state_dim=system.dim,
        param_dim=len(system.param_names),
        hidden_dim=hidden_dim,
    ).to(device)

    if model_path.exists():
        state_dict = torch.load(
            model_path, map_location=device, weights_only=True,
        )
        net.load_state_dict(state_dict)
        net.eval()
        logger.info("Loaded pre-trained coarse model from %s", model_path)
    else:
        logger.warning(
            "Pre-trained model not found at %s — falling back to "
            "lightweight training (run train_all.py first for best results)",
            model_path,
        )
        trainer = CoarseTrainer(system, device=device)
        net, _ = trainer.train(
            n_trajectories=100, epochs=1000, hidden_dim=hidden_dim,
        )

    return net


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

    slab_counts = [2, 4, 8, 12, 16]
    results = []

    for system_name in SYSTEM_REGISTRY:
        logger.info("-" * 40)
        logger.info("System: %s", system_name)
        
        system = get_system(system_name)

        # Load pre-trained coarse propagator
        model = _load_coarse_model(system, device=DEVICE)

        y0_cpu = system.default_initial_condition()
        y0_gpu = y0_cpu.to(DEVICE)
        params = system.default_params()
        theta = system.param_vector(params, device=DEVICE)

        # Serial reference: CPU is faster for sequential RK4
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
                trust_gate=TrustGate(lock_threshold=1e-5, lock_patience=1),
                max_iterations=30,
                system_name=system_name,
                n_workers=N_WORKERS,
                coarse_dt=0.1,
            )

            def _parareal_run():
                return parareal.solve(
                    f=system.f, y0=y0_gpu, t_span=system.default_time_span(),
                    n_slabs=n_slabs, fine_dt=0.01,
                    params=params, theta_ode=theta,
                    tolerance=1e-5, use_trust_gate=True,
                )

            result, elapsed = _timed_call(_parareal_run, DEVICE, n_runs=1)

            endpoint_err = torch.max(
                torch.abs(result.y[-1].cpu() - serial_result.y[-1].cpu())
            ).item()

            results.append({
                "system": system_name,
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

    fine_dt = 0.001  # 10× finer → 10× more serial work
    slab_counts = [2, 4, 8, 16]
    results = []

    for system_name in SYSTEM_REGISTRY:
        logger.info("-" * 40)
        logger.info("System: %s", system_name)
        
        system = get_system(system_name)

        # Load pre-trained coarse propagator
        model = _load_coarse_model(system, device=DEVICE)

        y0_cpu = system.default_initial_condition()
        y0_gpu = y0_cpu.to(DEVICE)
        params = system.default_params()
        theta = system.param_vector(params, device=DEVICE)

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
                trust_gate=TrustGate(lock_threshold=1e-5, lock_patience=1),
                max_iterations=50,
                system_name=system_name,
                n_workers=N_WORKERS,
                coarse_dt=0.1,
            )

            def _parareal_run():
                return parareal.solve(
                    f=system.f, y0=y0_gpu, t_span=system.default_time_span(),
                    n_slabs=n_slabs, fine_dt=fine_dt,
                    params=params, theta_ode=theta,
                    tolerance=1e-6, use_trust_gate=True,
                )

            result, elapsed = _timed_call(_parareal_run, DEVICE, n_runs=1)

            endpoint_err = torch.max(
                torch.abs(result.y[-1].cpu() - serial_result.y[-1].cpu())
            ).item()

            results.append({
                "system": system_name,
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
    """Run all benchmarks and save results.

    Supports three execution modes via ``--mode``:
        - ``cpu``: CPU-only, sequential fine pass (baseline).
        - ``gpu``: CUDA-accelerated coarse net, sequential fine pass.
        - ``multiproc``: CPU coarse net, multiprocessing fine pass.

    Results are saved to ``benchmarks/results/<mode>/``.
    """
    global DEVICE, N_WORKERS

    parser = argparse.ArgumentParser(
        description="Benchmark solvers in different execution modes.",
    )
    parser.add_argument(
        "--mode", type=str, default="gpu",
        choices=["cpu", "gpu", "multiproc"],
        help=(
            "Execution mode: "
            "'cpu' = CPU-only sequential, "
            "'gpu' = CUDA coarse + CPU fine, "
            "'multiproc' = CPU coarse + multiprocessing fine. "
            "(default: gpu)"
        ),
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of multiprocessing workers (multiproc mode only). "
             "Defaults to cpu_count//4.",
    )
    args = parser.parse_args()

    # -- Configure device and workers per mode -------------------------------
    if args.mode == "cpu":
        DEVICE = torch.device("cpu")
        N_WORKERS = 0
        mode_label = "CPU Sequential"
    elif args.mode == "gpu":
        if not torch.cuda.is_available():
            logger.error("CUDA not available — falling back to CPU mode")
            DEVICE = torch.device("cpu")
        else:
            DEVICE = torch.device("cuda")
        N_WORKERS = 0
        mode_label = (
            f"GPU ({torch.cuda.get_device_name(DEVICE)})"
            if DEVICE.type == "cuda" else "CPU (no CUDA)"
        )
    elif args.mode == "multiproc":
        DEVICE = torch.device("cpu")
        N_WORKERS = args.n_workers or (
            min(os.cpu_count() // 4, 16) if os.cpu_count() else 8
        )
        mode_label = f"CPU Multiprocessing ({N_WORKERS} workers)"

    logger.info("=" * 60)
    logger.info("BENCHMARK SUITE — Mode: %s", mode_label)
    logger.info("Device: %s | Workers: %d | PyTorch: %s",
                DEVICE, N_WORKERS, torch.__version__)
    logger.info("=" * 60)

    output_dir = Path(f"benchmarks/results/{args.mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark 1: Step sizes
    df_steps = benchmark_step_sizes()
    df_steps["mode"] = args.mode
    df_steps.to_csv(output_dir / "step_size_benchmark.csv", index=False)
    logger.info("Step size results saved to %s",
                output_dir / "step_size_benchmark.csv")
    print("\n" + df_steps.to_string(index=False))

    # Benchmark 2: Parareal slabs (easy — dt=0.01, 2K steps)
    print("\n")
    df_easy = benchmark_parareal_slabs()
    df_easy["mode"] = args.mode
    df_easy.to_csv(output_dir / "parareal_easy_benchmark.csv", index=False)
    logger.info("Parareal EASY results saved to %s",
                output_dir / "parareal_easy_benchmark.csv")
    print("\n" + df_easy.to_string(index=False))

    # Benchmark 3: Parareal slabs (hard — dt=0.001, 20K steps)
    print("\n")
    df_hard = benchmark_parareal_hard()
    df_hard["mode"] = args.mode
    df_hard.to_csv(output_dir / "parareal_hard_benchmark.csv", index=False)
    logger.info("Parareal HARD results saved to %s",
                output_dir / "parareal_hard_benchmark.csv")
    print("\n" + df_hard.to_string(index=False))

    logger.info("All benchmarks complete! Results in %s/", output_dir)


if __name__ == "__main__":
    main()
