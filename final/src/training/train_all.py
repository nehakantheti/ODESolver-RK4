"""End-to-end training orchestrator for all neural components.

Trains both the coarse propagator and k-factor residual networks for
a given ODE system, saves the trained models, and reports results.

This script can be run directly to train all models:
    py -3.11 -m src.training.train_all --system damped_oscillator

Or used programmatically:
    >>> from src.training.train_all import TrainingOrchestrator
    >>> orch = TrainingOrchestrator("damped_oscillator")
    >>> models = orch.train_all()
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure the 'final' directory is in sys.path to resolve 'src' imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from src.ode_systems import ODESystem, get_system, SYSTEM_REGISTRY
from src.networks.coarse_propagator import CoarsePropagatorNet
from src.networks.k_factor_residual import KFactorResidualNet
from src.training.train_coarse import CoarseTrainer
from src.training.train_k_factor import KFactorTrainer

logger = logging.getLogger(__name__)


@dataclass
class TrainedModels:
    """Container for trained model artifacts.

    Attributes:
        coarse_net: Trained coarse propagator model.
        k_factor_net: Trained k-factor residual model.
        coarse_history: Training loss history for coarse model.
        k_factor_history: Training loss history for k-factor model.
        system_name: Name of the ODE system these models were trained on.
    """
    coarse_net: CoarsePropagatorNet
    k_factor_net: KFactorResidualNet
    coarse_history: Dict
    k_factor_history: Dict
    system_name: str


class TrainingOrchestrator:
    """Orchestrates end-to-end training of all neural solver components.

    Manages the full pipeline: data generation, training of both networks,
    model saving, and result reporting.

    Attributes:
        system: The ODE system to train on.
        device: Torch device for computation.
        save_dir: Directory for saving trained model weights.

    Example:
        >>> orch = TrainingOrchestrator("damped_oscillator")
        >>> models = orch.train_all(coarse_epochs=3000, kfactor_epochs=2000)
        >>> # Models are trained and saved to trained_models/
    """

    def __init__(
        self,
        system_name: str,
        device: torch.device | None = None,
        save_dir: str | Path | None = None,
    ):
        """Initialise the training orchestrator.

        Args:
            system_name: Registry key for the ODE system.
            device: Torch device.  Auto-detects CUDA if available.
            save_dir: Directory to save trained model weights.
                     Defaults to ``trained_models/`` relative to the
                     project root (``final/``).
        """
        self.system = get_system(system_name)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Resolve save_dir relative to project root (final/)
        if save_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.save_dir = project_root / "trained_models"
        else:
            self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "TrainingOrchestrator: system='%s', device=%s, save_dir='%s'",
            system_name, self.device, self.save_dir,
        )
        if self.device.type == "cuda":
            gpu_props = torch.cuda.get_device_properties(self.device)
            logger.info(
                "GPU: %s | VRAM: %.1f GB | Compute: %d.%d",
                gpu_props.name, gpu_props.total_memory / 1e9,
                gpu_props.major, gpu_props.minor,
            )

    def train_all(
        self,
        n_trajectories: int = 200,
        coarse_epochs: int = 1500,
        kfactor_epochs: int = 500,
        coarse_hidden: int = 128,
        kfactor_hidden: int = 96,
        coarse_dt: float = 0.1,
        fine_dt: float = 0.001,
        kfactor_dt: float = 0.01,
        lbfgs_steps: int = 50,
    ) -> TrainedModels:
        """Train all neural components end-to-end.

        Steps:
            1. Train the coarse propagator network.
            2. Train the k-factor residual network.
            3. Save both models to disk.
            4. Return the trained models and histories.

        Args:
            n_trajectories: Number of trajectories for data generation.
            coarse_epochs: Training epochs for the coarse propagator.
            kfactor_epochs: Training epochs for the k-factor network.
            coarse_hidden: Hidden dimension for the coarse network.
            kfactor_hidden: Hidden dimension for the k-factor network.
            coarse_dt: Coarse time step for the coarse propagator.
            fine_dt: Fine RK4 step for ground truth generation.
            kfactor_dt: Step size for k-factor data generation.
            lbfgs_steps: Number of L-BFGS fine-tuning steps after Adam
                        for both networks.  Set to 0 to disable.

        Returns:
            ``TrainedModels`` containing both trained models.
        """
        total_start = time.time()
        system_name = self.system.name.lower().replace(" ", "_")

        # -- Train coarse propagator ----------------------------------------
        logger.info("=" * 60)
        logger.info("TRAINING COARSE PROPAGATOR")
        logger.info("=" * 60)

        coarse_trainer = CoarseTrainer(
            system=self.system, device=self.device,
        )
        coarse_net, coarse_history = coarse_trainer.train(
            n_trajectories=n_trajectories,
            fine_dt=fine_dt,
            coarse_dt=coarse_dt,
            epochs=coarse_epochs,
            hidden_dim=coarse_hidden,
            lbfgs_steps=lbfgs_steps,
        )

        # Free GPU memory before training the next model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared before k-factor training")

        # Save coarse model
        coarse_path = self.save_dir / f"coarse_{system_name}.pt"
        torch.save(coarse_net.state_dict(), coarse_path)
        logger.info("Coarse model saved to %s", coarse_path)

        # -- Train k-factor network -----------------------------------------
        logger.info("=" * 60)
        logger.info("TRAINING K-FACTOR RESIDUAL NETWORK")
        logger.info("=" * 60)

        kfactor_trainer = KFactorTrainer(
            system=self.system, device=self.device,
        )
        kfactor_net, kfactor_history = kfactor_trainer.train(
            n_trajectories=n_trajectories,
            dt=kfactor_dt,
            epochs=kfactor_epochs,
            hidden_dim=kfactor_hidden,
            lbfgs_steps=lbfgs_steps,
        )

        # Save k-factor model
        kfactor_path = self.save_dir / f"kfactor_{system_name}.pt"
        torch.save(kfactor_net.state_dict(), kfactor_path)
        logger.info("K-factor model saved to %s", kfactor_path)

        # -- Summary --------------------------------------------------------
        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("Total time: %.1f seconds (%.1f minutes)",
                     total_time, total_time / 60)
        logger.info("Coarse final loss: train=%.6f, val=%.6f",
                     coarse_history["train_loss"][-1],
                     coarse_history["val_loss"][-1])
        logger.info("K-factor final loss: train=%.6f, val=%.6f",
                     kfactor_history["train_loss"][-1],
                     kfactor_history["val_loss"][-1])
        logger.info("=" * 60)

        return TrainedModels(
            coarse_net=coarse_net,
            k_factor_net=kfactor_net,
            coarse_history=coarse_history,
            k_factor_history=kfactor_history,
            system_name=system_name,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for training all models."""
    parser = argparse.ArgumentParser(
        description="Train neural components for the parallel RK4 solver."
    )
    parser.add_argument(
        "--system", type=str, default="damped_oscillator",
        choices=list(SYSTEM_REGISTRY.keys()),
        help="ODE system to train on.",
    )
    parser.add_argument(
        "--coarse-epochs", type=int, default=5000,
        help="Training epochs for the coarse propagator.",
    )
    parser.add_argument(
        "--kfactor-epochs", type=int, default=3000,
        help="Training epochs for the k-factor network.",
    )
    parser.add_argument(
        "--n-trajectories", type=int, default=200,
        help="Number of diverse trajectories for data generation.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda"],
        help="Force training device. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--lbfgs-steps", type=int, default=50,
        help="L-BFGS fine-tuning steps after Adam (0 to disable).",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device) if args.device else None
    orch = TrainingOrchestrator(args.system, device=device)
    orch.train_all(
        n_trajectories=args.n_trajectories,
        coarse_epochs=args.coarse_epochs,
        kfactor_epochs=args.kfactor_epochs,
        lbfgs_steps=args.lbfgs_steps,
    )


if __name__ == "__main__":
    main()
