"""Train Neural Coarse Propagators across all ODE systems.

This script trains the CoarsePropagatorNet for either a specific ODE system
or all systems in the registry. It is configured for high-resolution
inference (coarse_dt=0.01) to ensure Parareal iteration counts remain
extremely low (K ≈ 2-3).

Usage:
    # Train all systems sequentially
    python train_all.py

    # Train specific systems (useful for multi-GPU distribution)
    CUDA_VISIBLE_DEVICES=0 python train_all.py --system damped_oscillator
    CUDA_VISIBLE_DEVICES=1 python train_all.py --system lotka_volterra
    CUDA_VISIBLE_DEVICES=2 python train_all.py --system van_der_pol
"""

import argparse
import logging
import os
import torch
from pathlib import Path

from src.ode_systems import get_system, SYSTEM_REGISTRY
from src.training.train_coarse import CoarseTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Neural Coarse Propagators.")
    parser.add_argument(
        "--system", type=str, default="all",
        choices=["all"] + list(SYSTEM_REGISTRY.keys()),
        help="System to train, or 'all' to train sequentially."
    )
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--coarse-dt", type=float, default=0.1, help="Base coarse dt")
    args = parser.parse_args()

    # Determine systems to train
    systems_to_train = list(SYSTEM_REGISTRY.keys()) if args.system == "all" else [args.system]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting High-Resolution Coarse Training on device: %s", device)

    # Ensure output directory exists
    output_dir = Path("trained_models")
    output_dir.mkdir(exist_ok=True)

    for sys_name in systems_to_train:
        logger.info("=" * 60)
        logger.info("TRAINING SYSTEM: %s", sys_name.upper())
        logger.info("=" * 60)

        system = get_system(sys_name)
        trainer = CoarseTrainer(system, device=device)

        # Train with optimal Phase 11 parameters
        # randomize_dt=True and coarse_dt=0.1 will sample dt in [0.05, 0.2]
        model, history = trainer.train(
            n_trajectories=300,        # Lots of data for accuracy
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            coarse_dt=args.coarse_dt,
            randomize_dt=True,
            physics_weight=0.01,
            lbfgs_steps=100,           # Strong L-BFGS polish
        )

        # Save model
        sys_key = system.name.lower().replace(" ", "_")
        model_path = output_dir / f"coarse_{sys_key}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info("Successfully saved model to %s", model_path)

    logger.info("All requested training completed successfully!")


if __name__ == "__main__":
    main()
