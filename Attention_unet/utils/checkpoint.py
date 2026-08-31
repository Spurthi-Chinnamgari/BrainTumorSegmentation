"""
Model Checkpointing and Persistence Utilities.
Manages saving and loading model checkpoints, optimizer state, and best metric tracking.
"""

from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
    best_filename: str = "best_model.pth",
) -> str:
    """
    Save PyTorch checkpoint dictionary to directory.

    Args:
        state: Dictionary containing 'model_state_dict', 'optimizer_state_dict', 'epoch', etc.
        checkpoint_dir: Directory path to save checkpoint.
        filename: Checkpoint filename.
        is_best: If True, copies checkpoint to best_filename.
        best_filename: Best model filename.
    Returns:
        Path to saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_filepath)

    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load PyTorch checkpoint file into model, optimizer, and scheduler.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        model: PyTorch model instance.
        optimizer: Optional optimizer instance.
        scheduler: Optional learning rate scheduler instance.
        device: Torch device ("cpu" or "cuda").
    Returns:
        Checkpoint dictionary loaded from disk.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class CheckpointManager:
    """
    Tracks validation metrics, manages automatic checkpoint saving, and best model preservation.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = "val_dice",
        mode: str = "max",
    ):
        """
        Args:
            checkpoint_dir: Directory to save models.
            metric_name: Validation metric monitored for best model saving.
            mode: "max" (higher metric is better) or "min" (lower metric is better).
        """
        self.checkpoint_dir = checkpoint_dir
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")

    def is_improvement(self, current_metric: float) -> bool:
        if self.mode == "max":
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric

    def step(
        self,
        current_metric: float,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate current metric, save latest checkpoint, and update best model if improved.

        Returns:
            True if current metric is a new best, False otherwise.
        """
        is_best = self.is_improvement(current_metric)
        if is_best:
            self.best_metric = current_metric

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": self.best_metric,
            "metric_name": self.metric_name,
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if extra_state is not None:
            state.update(extra_state)

        # Save latest checkpoint
        save_checkpoint(
            state,
            checkpoint_dir=self.checkpoint_dir,
            filename="latest_checkpoint.pth",
            is_best=is_best,
            best_filename="best_model.pth",
        )

        return is_best
