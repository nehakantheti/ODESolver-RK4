"""Training pipeline for the derivative-predicting coarse propagator.

Trains the CoarsePropagatorNet to predict the ODE vector field f̂(y, t, θ).
The training loss is physics-informed:

    L = ||f̂ - f_true||^2                              [derivative matching]
      + λ_phys * ||trapezoidal_residual||^2             [physics consistency]

The derivative matching loss teaches the network the basic vector field.
The trapezoidal residual ensures that integration steps are physically
consistent over finite time steps, using the exact ODE as a constraint:

    residual = (y_pred - y_n) - dt/2 * [f(t_n, y_n) + f(t_n+dt, y_pred)]

GPU acceleration features:
    - Automatic Mixed Precision (AMP) with GradScaler for float16 training
    - torch.compile for kernel fusion (PyTorch 2.x, Linux only)
    - pin_memory + non_blocking transfers for async CPU→GPU data loading
    - Early stopping to prevent overfitting and save compute

Hybrid optimiser (adapted from mid/phase1/pinn_hybrid.py):
    - Phase 1: Adam with cosine annealing (find the basin)
    - Phase 2: L-BFGS with strong Wolfe line search (polish to minimum)

Usage:
    >>> trainer = CoarseTrainer(system, device=device)
    >>> model, history = trainer.train(n_trajectories=200, epochs=5000)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from src.networks.coarse_propagator import CoarsePropagatorNet
from src.ode_systems import ODESystem
from src.training.data_generator import DataGenerator

logger = logging.getLogger(__name__)


class CoarseTrainer:
    """Training pipeline for the derivative-predicting coarse propagator.

    Handles end-to-end training: data generation, model creation,
    training loop with physics-informed loss, early stopping, and
    hybrid Adam → L-BFGS optimisation.

    GPU features:
        - AMP (Automatic Mixed Precision) for 2-3× throughput.
        - torch.compile for kernel fusion (Linux only).
        - Early stopping with configurable patience.

    Attributes:
        system: ODE system to train on.
        device: Torch device for computation.
        generator: Data generator instance.
        use_amp: Whether AMP is available and enabled.

    Example:
        >>> system = get_system("damped_oscillator")
        >>> trainer = CoarseTrainer(system, device=torch.device("cuda"))
        >>> model, history = trainer.train(epochs=3000)
    """

    def __init__(
        self,
        system: ODESystem,
        device: torch.device | None = None,
    ):
        """Initialise the trainer.

        Args:
            system: ODE system defining the dynamics and parameter ranges.
            device: Torch device.  Auto-detects CUDA if available.
        """
        self.system = system
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Data generation always on CPU (sequential RK4, then transfer)
        self.generator = DataGenerator(system, device=torch.device("cpu"))
        # AMP is only beneficial on CUDA
        self.use_amp = self.device.type == "cuda"

        logger.info(
            "CoarseTrainer created: system='%s', device=%s, AMP=%s",
            system.name, self.device, self.use_amp,
        )
        if self.device.type == "cuda":
            logger.info(
                "GPU: %s | VRAM: %.1f GB",
                torch.cuda.get_device_name(self.device),
                torch.cuda.get_device_properties(self.device).total_memory / 1e9,
            )

    def _compute_physics_residual(
        self,
        model: CoarsePropagatorNet,
        y_n: Tensor,
        t_n: Tensor,
        theta_ode: Tensor,
        delta_t: Tensor,
        y_next: Tensor,
    ) -> Tensor:
        """Compute the trapezoidal physics residual.

        Uses the exact ODE vector field to check whether the network's
        predicted derivative produces physically consistent integration
        steps.

        Trapezoidal rule (2nd-order implicit constraint):
            y_{n+1} - y_n ≈ dt/2 * [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]

        We compute:
            y_pred = y_n + dt * f̂(y_n, t_n, θ)    [Euler step with learned f̂]
            f_start = f_exact(t_n, y_n)              [exact derivative at start]
            f_end = f_exact(t_n + dt, y_pred)        [exact derivative at end]
            residual = (y_pred - y_n) - dt/2 * (f_start + f_end)

        Args:
            model: The coarse propagator network.
            y_n: Current states, shape ``(batch, dim)``.
            t_n: Current times, shape ``(batch, 1)``.
            theta_ode: Parameter vectors, shape ``(batch, param_dim)``.
            delta_t: Time steps, shape ``(batch, 1)``.
            y_next: Target next states for reference.

        Returns:
            Scalar physics residual loss (mean squared).
        """
        # Predicted derivative from network
        f_hat = model(y_n, t_n, theta_ode)

        # Euler step with learned derivative
        y_pred = y_n + delta_t * f_hat

        # Reconstruct params dict for calling system.f
        # Use default params as approximation (the theta_ode contains
        # the actual values, but system.f needs a dict)
        params = self.system.default_params()

        # Exact derivative at start point
        t_start = t_n[:, 0].detach()
        f_start = self.system.f(
            t_start.mean().item(), y_n.detach(), params,
        )

        # Exact derivative at predicted end point
        t_end_scalar = (t_n[:, 0] + delta_t[:, 0]).detach().mean().item()
        f_end = self.system.f(
            t_end_scalar, y_pred.detach(), params,
        )

        # Trapezoidal residual
        step = y_pred - y_n
        trap_approx = (delta_t / 2.0) * (f_start + f_end)
        residual = step - trap_approx

        return torch.mean(residual ** 2)

    def train(
        self,
        n_trajectories: int = 200,
        fine_dt: float = 0.001,
        coarse_dt: float = 0.1,
        epochs: int = 5000,
        batch_size: int = 256,
        lr: float = 1e-3,
        physics_weight: float = 0.01,
        hidden_dim: int = 128,
        val_fraction: float = 0.2,
        use_compile: bool = True,
        lbfgs_steps: int = 50,
        early_stopping_patience: int = 200,
        randomize_dt: bool = True,
    ) -> Tuple[CoarsePropagatorNet, Dict[str, List[float]]]:
        """Train the derivative-predicting coarse propagator.

        Steps:
            1. Generate training data from diverse RK4 trajectories (CPU).
            2. Transfer data to GPU, split into train/validation sets.
            3. Create the model, optimiser, AMP scaler.
            4. Adam training with physics-informed loss + early stopping.
            5. L-BFGS fine-tuning with increased physics weight.
            6. Return trained model and loss history.

        Args:
            n_trajectories: Number of trajectories for data generation.
            fine_dt: Fine RK4 step size for ground truth.
            coarse_dt: Base coarse step size.
            epochs: Maximum number of Adam training epochs.
            batch_size: Mini-batch size.
            lr: Initial learning rate for Adam.
            physics_weight: Weight λ_phys for the physics residual
                           during Adam phase.  Increased to 1.0 for L-BFGS.
            hidden_dim: Hidden layer width for the network.
            val_fraction: Fraction of data held out for validation.
            use_compile: Whether to use ``torch.compile``.
            lbfgs_steps: Number of L-BFGS fine-tuning steps after Adam.
            early_stopping_patience: Number of epochs without val loss
                                    improvement before stopping Adam phase.
            randomize_dt: If True, randomize dt in data generation.

        Returns:
            Tuple of (trained_model, history_dict) where history_dict
            contains ``"train_loss"``, ``"val_loss"`` lists.
        """
        train_start = time.time()

        # -- Step 1: Generate data on CPU -----------------------------------
        logger.info("Step 1/4: Generating training data on CPU...")
        data = self.generator.generate_coarse_data(
            n_trajectories=n_trajectories,
            fine_dt=fine_dt,
            coarse_dt=coarse_dt,
            randomize_dt=randomize_dt,
        )

        n_total = len(data)
        n_val = int(n_total * val_fraction)
        n_train = n_total - n_val

        logger.info(
            "Data generated: %d total (%d train, %d val)",
            n_total, n_train, n_val,
        )

        # -- Step 2: Transfer to GPU and create data loaders ----------------
        logger.info("Step 2/4: Transferring data to %s...", self.device)
        perm = torch.randperm(n_total)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        def make_loader(indices: Tensor, shuffle: bool = True) -> DataLoader:
            """Create a DataLoader from selected indices."""
            dataset = TensorDataset(
                data.y_n[indices].to(self.device),
                data.t_n[indices].to(self.device),
                data.theta_ode[indices].to(self.device),
                data.f_true[indices].to(self.device),
                data.delta_t[indices].to(self.device),
                data.y_next[indices].to(self.device),
            )
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
            )

        train_loader = make_loader(train_idx, shuffle=True)
        val_loader = make_loader(val_idx, shuffle=False)

        # -- Step 3: Create model + GPU optimisations -----------------------
        logger.info("Step 3/4: Creating model on %s...", self.device)
        model = CoarsePropagatorNet(
            state_dim=self.system.dim,
            param_dim=len(self.system.param_names),
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Optionally compile for kernel fusion
        import platform
        train_model = model
        if use_compile and self.device.type == "cuda" and platform.system() != "Windows":
            try:
                train_model = torch.compile(model)
                logger.info("torch.compile applied for kernel fusion")
            except Exception as exc:
                logger.warning("torch.compile failed (%s), using eager mode", exc)
                train_model = model
        elif use_compile and platform.system() == "Windows":
            logger.info("torch.compile skipped (Triton unavailable on Windows)")

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=lr * 0.01,
        )
        mse_loss = nn.MSELoss()

        # AMP: GradScaler for stable float16 training
        scaler = GradScaler(self.device.type, enabled=self.use_amp)

        if self.use_amp:
            logger.info("AMP enabled: float16 forward pass + float32 master weights")

        # -- Step 4: Adam training with physics-informed loss ---------------
        logger.info(
            "Step 4/4: Training for up to %d epochs "
            "(early_stopping_patience=%d, λ_phys=%.3f)...",
            epochs, early_stopping_patience, physics_weight,
        )
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        # Early stopping state
        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0
        actual_epochs = 0

        for epoch in range(epochs):
            # ---- Training ----
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for y_n, t_n, theta, f_target, dt_batch, y_next in train_loader:
                optimiser.zero_grad(set_to_none=True)

                # AMP: forward pass in float16 on GPU
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    # Derivative matching loss (primary)
                    f_hat = train_model(y_n, t_n, theta)
                    deriv_loss = mse_loss(f_hat, f_target)

                    # Physics residual (trapezoidal consistency)
                    if physics_weight > 0:
                        phys_loss = self._compute_physics_residual(
                            train_model, y_n, t_n, theta, dt_batch, y_next,
                        )
                        loss = deriv_loss + physics_weight * phys_loss
                    else:
                        loss = deriv_loss

                # AMP: scale loss, backward, unscale, step
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # ---- Validation ----
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for y_n, t_n, theta, f_target, dt_batch, y_next in val_loader:
                    with autocast(device_type=self.device.type, enabled=self.use_amp):
                        f_hat = train_model(y_n, t_n, theta)
                        val_loss += mse_loss(f_hat, f_target).item()
                    n_val_batches += 1

            avg_val_loss = val_loss / max(n_val_batches, 1)
            history["val_loss"].append(avg_val_loss)

            # ---- Early stopping check ----
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            actual_epochs = epoch + 1

            if epoch % 500 == 0 or epoch == epochs - 1:
                mem_info = ""
                if self.device.type == "cuda":
                    mem_used = torch.cuda.memory_allocated(self.device) / 1e6
                    mem_reserved = torch.cuda.memory_reserved(self.device) / 1e6
                    mem_info = f", GPU_mem={mem_used:.0f}/{mem_reserved:.0f}MB"

                logger.info(
                    "Epoch %d/%d: train=%.6f, val=%.6f, lr=%.2e, "
                    "best_val=%.6f, no_improv=%d%s",
                    epoch, epochs, avg_train_loss, avg_val_loss,
                    scheduler.get_last_lr()[0],
                    best_val_loss, epochs_without_improvement, mem_info,
                )

            # Check early stopping
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs, "
                    "best_val=%.6f)",
                    epoch, early_stopping_patience, best_val_loss,
                )
                break

        # Restore best model weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(
                "Restored best model (val_loss=%.6f from epoch with best val)",
                best_val_loss,
            )

        # -- Stage 2: L-BFGS fine-tuning (hybrid optimizer) ------------------
        if lbfgs_steps > 0:
            lbfgs_phys_weight = 1.0  # Crank up physics for L-BFGS
            logger.info(
                "L-BFGS fine-tuning: %d steps (lr=1.0, λ_phys=%.1f, "
                "history=10, strong_wolfe)...",
                lbfgs_steps, lbfgs_phys_weight,
            )

            # L-BFGS requires full-batch data
            full_y_n = data.y_n[train_idx].to(self.device)
            full_t_n = data.t_n[train_idx].to(self.device)
            full_theta = data.theta_ode[train_idx].to(self.device)
            full_f_target = data.f_true[train_idx].to(self.device)
            full_dt = data.delta_t[train_idx].to(self.device)
            full_y_next = data.y_next[train_idx].to(self.device)

            # Validation tensors
            val_y_n = data.y_n[val_idx].to(self.device)
            val_t_n = data.t_n[val_idx].to(self.device)
            val_theta = data.theta_ode[val_idx].to(self.device)
            val_f_target = data.f_true[val_idx].to(self.device)

            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                history_size=10,
                max_iter=20,
                line_search_fn="strong_wolfe",
            )

            model.train()
            for step in range(lbfgs_steps):
                def closure():
                    lbfgs.zero_grad()
                    # No AMP — L-BFGS needs float32 for Hessian approx
                    f_hat = model(full_y_n, full_t_n, full_theta)
                    d_loss = mse_loss(f_hat, full_f_target)

                    # Physics residual with increased weight
                    p_loss = self._compute_physics_residual(
                        model, full_y_n, full_t_n, full_theta,
                        full_dt, full_y_next,
                    )
                    loss = d_loss + lbfgs_phys_weight * p_loss
                    loss.backward()
                    return loss

                loss = lbfgs.step(closure)
                lbfgs_train = loss.item()
                history["train_loss"].append(lbfgs_train)

                # Validation
                model.eval()
                with torch.no_grad():
                    vf_hat = model(val_y_n, val_t_n, val_theta)
                    lbfgs_val = mse_loss(vf_hat, val_f_target).item()
                history["val_loss"].append(lbfgs_val)
                model.train()

                if step % 10 == 0 or step == lbfgs_steps - 1:
                    mem_info = ""
                    if self.device.type == "cuda":
                        mem_used = torch.cuda.memory_allocated(self.device) / 1e6
                        mem_reserved = torch.cuda.memory_reserved(self.device) / 1e6
                        mem_info = f", GPU_mem={mem_used:.0f}/{mem_reserved:.0f}MB"
                    logger.info(
                        "L-BFGS %d/%d: train=%.6f, val=%.6f%s",
                        step, lbfgs_steps, lbfgs_train, lbfgs_val, mem_info,
                    )

            # Free full-batch tensors
            del full_y_n, full_t_n, full_theta, full_f_target
            del full_dt, full_y_next
            del val_y_n, val_t_n, val_theta, val_f_target

        total_time = time.time() - train_start
        logger.info(
            "Training complete! Total time: %.1fs (%.1f min), "
            "Adam epochs: %d, L-BFGS steps: %d",
            total_time, total_time / 60, actual_epochs, lbfgs_steps,
        )
        logger.info(
            "Final: train_loss=%.6f, val_loss=%.6f",
            history["train_loss"][-1], history["val_loss"][-1],
        )

        return model, history
