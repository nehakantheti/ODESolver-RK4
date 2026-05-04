"""Coarse propagator comparison benchmark.

Compares three coarse propagators in the Parareal pipeline:
    1. Euler (baseline)
    2. Backward Euler (stable baseline)
    3. Neural Network (derivative-predicting, trained)

Metrics:
    - Iterations to converge (K)
    - Total fine RK4 solves
    - Wall time
    - Final error vs sequential RK4
    - Speedup ratio vs sequential RK4

Usage:
    cd /home/neha/projects/ODE-RK/final
    python benchmarks/benchmark_coarse_comparison.py

    Optional flags:
        --system damped_oscillator|lotka_volterra|van_der_pol|lorenz
        --n-slabs 8 16
        --model-path trained_models/coarse_damped_harmonic_oscillator.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ode_systems import get_system, SYSTEM_REGISTRY
from src.networks.coarse_propagator import CoarsePropagatorNet
from src.networks.trust_gate import TrustGate
from src.solvers.classical_rk4 import ClassicalRK4Solver
from src.solvers.parareal import PararealSolver

logger = logging.getLogger(__name__)


def run_sequential_rk4(system, params, y0, t_span, fine_dt):
    """Run sequential RK4 as the ground-truth reference."""
    solver = ClassicalRK4Solver(device=torch.device("cpu"))
    start = time.perf_counter()
    result = solver.solve_single(
        f=system.f, y0=y0, t_span=t_span, dt=fine_dt, params=params,
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_parareal(
    system, params, y0, t_span, fine_dt, n_slabs, coarse_mode,
    coarse_net=None, theta_ode=None, coarse_dt=0.1,
    tolerance=1e-6, max_iter=50, **kwargs
):
    """Run Parareal with a specified coarse propagator."""
    solver = PararealSolver(
        coarse_net=coarse_net,
        device=torch.device("cpu"),
        trust_gate=TrustGate(lock_threshold=1e-5, lock_patience=1),
        max_iterations=max_iter,
        coarse_dt=coarse_dt,
        coarse_mode=coarse_mode,
        system=system,
        n_workers=kwargs.get("n_workers", 0),
    )

    result = solver.solve(
        f=system.f,
        y0=y0,
        t_span=t_span,
        n_slabs=n_slabs,
        fine_dt=fine_dt,
        params=params,
        theta_ode=theta_ode,
        tolerance=tolerance,
        use_trust_gate=True,
    )

    return result


def compute_error(parareal_result, rk4_result, n_slabs, t_span, fine_dt):
    """Compute max error between Parareal and sequential RK4 at slab boundaries."""
    delta_t = (t_span[1] - t_span[0]) / n_slabs
    max_error = 0.0

    for n in range(n_slabs + 1):
        slab_time = t_span[0] + n * delta_t
        serial_idx = int(round((slab_time - t_span[0]) / fine_dt))
        serial_idx = min(serial_idx, rk4_result.y.shape[0] - 1)

        diff = torch.max(
            torch.abs(parareal_result.y[n] - rk4_result.y[serial_idx])
        ).item()
        max_error = max(max_error, diff)

    return max_error


def load_coarse_net(model_path, system):
    """Load a trained coarse propagator network."""
    if not os.path.exists(model_path):
        logger.warning("Model not found at %s, will skip neural mode", model_path)
        return None

    net = CoarsePropagatorNet(
        state_dim=system.dim,
        param_dim=len(system.param_names),
        hidden_dim=128,
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Handle potential key mismatches from old model format
    try:
        net.load_state_dict(state_dict, strict=False)
        logger.info("Loaded coarse model from %s", model_path)
    except Exception as e:
        logger.warning("Failed to load model from %s: %s", model_path, e)
        return None

    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser(
        description="Compare coarse propagators in Parareal pipeline."
    )
    parser.add_argument(
        "--system", type=str, default="damped_oscillator",
        choices=list(SYSTEM_REGISTRY.keys()),
    )
    parser.add_argument("--n-slabs", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--fine-dt", type=float, default=0.001)
    parser.add_argument("--coarse-dt", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--n-workers", type=int, default=0, help="Number of workers for multiprocessing fine pass")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    system = get_system(args.system)
    params = system.default_params()
    y0 = system.default_initial_condition()
    t_span = system.default_time_span()
    theta = system.param_vector(params)

    # Resolve model path
    if args.model_path is None:
        sys_key = system.name.lower().replace(" ", "_")
        model_dir = Path(__file__).resolve().parent.parent / "trained_models"
        args.model_path = str(model_dir / f"coarse_{sys_key}.pt")

    coarse_net = load_coarse_net(args.model_path, system)

    # 1. Sequential RK4 baseline
    logger.info("=" * 70)
    logger.info("SEQUENTIAL RK4 BASELINE")
    logger.info("=" * 70)
    rk4_result, rk4_time = run_sequential_rk4(
        system, params, y0, t_span, args.fine_dt,
    )
    logger.info(
        "Sequential RK4: %d steps, time=%.3fs",
        rk4_result.y.shape[0] - 1, rk4_time,
    )

    # 2. Parareal with different coarse propagators
    all_results = []

    coarse_modes = ["euler", "rk2", "backward_euler"]
    if coarse_net is not None:
        coarse_modes.append("neural")

    for n_slabs in args.n_slabs:
        for mode in coarse_modes:
            logger.info("=" * 70)
            logger.info(
                "PARAREAL: coarse=%s, P=%d, tol=%.1e",
                mode.upper(), n_slabs, args.tolerance,
            )
            logger.info("=" * 70)

            try:
                result = run_parareal(
                    system=system,
                    params=params,
                    y0=y0,
                    t_span=t_span,
                    fine_dt=args.fine_dt,
                    n_slabs=n_slabs,
                    coarse_mode=mode,
                    coarse_net=coarse_net if mode == "neural" else None,
                    theta_ode=theta,
                    coarse_dt=args.coarse_dt,
                    tolerance=args.tolerance,
                    max_iter=args.max_iter,
                    n_workers=args.n_workers,
                )

                error = compute_error(
                    result, rk4_result, n_slabs, t_span, args.fine_dt,
                )
                converged = (
                    result.convergence_history[-1] < args.tolerance
                    if result.convergence_history else False
                )
                total_fine = sum(result.fine_solves_per_iter)
                speedup = rk4_time / max(result.wall_time, 1e-9)

                entry = {
                    "system": args.system,
                    "coarse_mode": mode,
                    "n_slabs": n_slabs,
                    "K_iterations": result.n_iterations,
                    "converged": converged,
                    "total_fine_solves": total_fine,
                    "wall_time_s": round(result.wall_time, 4),
                    "rk4_time_s": round(rk4_time, 4),
                    "speedup": round(speedup, 3),
                    "max_error_vs_rk4": f"{error:.2e}",
                    "final_change": f"{result.convergence_history[-1]:.2e}"
                    if result.convergence_history else "N/A",
                }
                all_results.append(entry)

                logger.info(
                    "Result: K=%d, converged=%s, fine_solves=%d, "
                    "time=%.3fs, speedup=%.2f×, error=%.2e",
                    result.n_iterations, converged, total_fine,
                    result.wall_time, speedup, error,
                )

            except Exception as e:
                logger.error("FAILED: %s (n_slabs=%d): %s", mode, n_slabs, e)
                all_results.append({
                    "system": args.system,
                    "coarse_mode": mode,
                    "n_slabs": n_slabs,
                    "error": str(e),
                })

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'System':<22} {'Coarse':<16} {'P':>3} {'K':>4} {'Conv':>5} "
          f"{'Fine':>6} {'Time(s)':>8} {'Speedup':>8} {'MaxErr':>10}")
    print("=" * 90)
    for r in all_results:
        if "error" in r:
            print(f"{r['system']:<22} {r['coarse_mode']:<16} "
                  f"{r['n_slabs']:>3}  FAILED: {r['error']}")
        else:
            print(
                f"{r['system']:<22} {r['coarse_mode']:<16} "
                f"{r['n_slabs']:>3} {r['K_iterations']:>4} "
                f"{'✓' if r['converged'] else '✗':>5} "
                f"{r['total_fine_solves']:>6} "
                f"{r['wall_time_s']:>8.3f} "
                f"{r['speedup']:>7.2f}× "
                f"{r['max_error_vs_rk4']:>10}"
            )
    print("=" * 90)
    print(f"Sequential RK4 baseline: {rk4_time:.3f}s")

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / f"coarse_comparison_{args.system}.json"
    with open(out_file, "w") as fp:
        json.dump(all_results, fp, indent=2)
    logger.info("Results saved to %s", out_file)


if __name__ == "__main__":
    main()
