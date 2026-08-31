"""
Configuration module for 3D Attention U-Net
Multimodal Brain Tumor Segmentation (BraTS 2023).
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os
import torch


# ============================================================
# DATASET CONFIGURATION
# ============================================================

@dataclass
class DatasetConfig:
    """Dataset and preprocessing configuration."""

    # --------------------------------------------------------
    # BraTS 2023 dataset path
    # --------------------------------------------------------

    data_dir: str = (
        r"C:\Datasets\BraTS2023"
        r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )

    train_dir: str = (
        r"C:\Datasets\BraTS2023"
        r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )

    val_dir: str = (
        r"C:\Datasets\BraTS2023"
        r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )

    test_dir: str = (
        r"C:\Datasets\BraTS2023"
        r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )

    # --------------------------------------------------------
    # MRI modalities
    #
    # t1n = T1 native
    # t1c = T1 contrast enhanced
    # t2w = T2 weighted
    # t2f = T2 FLAIR
    # --------------------------------------------------------

    modalities: List[str] = field(
        default_factory=lambda: [
            "t1n",
            "t1c",
            "t2w",
            "t2f",
        ]
    )

    num_modalities: int = 4

    in_channels: int = 4

    # --------------------------------------------------------
    # Segmentation classes
    #
    # 0 = Background
    # 1 = NCR
    # 2 = Edema
    # 3 = Enhancing Tumor
    # --------------------------------------------------------

    num_classes: int = 4

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This MUST match the patch used by train.py.
    #
    # Current training patch:
    #       (16, 32, 32)
    #
    # Do NOT put (128,128,128) here.
    # --------------------------------------------------------

    patch_size_3d: Tuple[int, int, int] = (
        16,
        32,
        32,
    )

    # --------------------------------------------------------
    # 2D configuration
    # --------------------------------------------------------

    image_size_2d: Tuple[int, int] = (
        32,
        32,
    )

    slice_dim: int = 2

    # --------------------------------------------------------
    # Dataset splitting
    # --------------------------------------------------------

    val_split: float = 0.15

    test_split: float = 0.15

    seed: int = 42

    # --------------------------------------------------------
    # Intensity normalization
    # --------------------------------------------------------

    normalize_intensity: bool = True

    z_score: bool = True


# ============================================================
# MODEL CONFIGURATION
# ============================================================

@dataclass
class ModelConfig:
    """Attention U-Net model configuration."""

    # --------------------------------------------------------
    # We are using 3D
    # --------------------------------------------------------

    dimension: str = "3d"

    # --------------------------------------------------------
    # Input / output
    # --------------------------------------------------------

    in_channels: int = 4

    out_channels: int = 4

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # These MUST match the model used during training.
    #
    # Your train.py currently uses:
    #
    # [16, 32, 64, 128, 256]
    # --------------------------------------------------------

    features: List[int] = field(
        default_factory=lambda: [
            16,
            32,
            64,
            128,
            256,
        ]
    )

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    dropout: float = 0.1

    # --------------------------------------------------------
    # Model options
    # --------------------------------------------------------

    use_batch_norm: bool = True

    activation: str = "relu"

    use_transpose: bool = True


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

@dataclass
class TrainConfig:
    """Training, optimizer and logging configuration."""

    # --------------------------------------------------------
    # Training dimension
    # --------------------------------------------------------

    dimension: str = "3d"

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    batch_size: int = 2

    val_batch_size: int = 2

    # --------------------------------------------------------
    # Number of epochs
    # --------------------------------------------------------

    epochs: int = 100

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    learning_rate: float = 1e-4

    min_lr: float = 1e-6

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer: str = "adamw"

    weight_decay: float = 1e-5

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler: str = "cosine"

    step_size: int = 30

    gamma: float = 0.5

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping_patience: int = 15

    # --------------------------------------------------------
    # Automatic Mixed Precision
    # --------------------------------------------------------

    use_amp: bool = True

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # DataLoader workers
    #
    # Keeping this 0 avoids Windows multiprocessing problems.
    # --------------------------------------------------------

    num_workers: int = 0

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    checkpoint_dir: str = "./checkpoints"

    log_dir: str = "./logs"

    results_dir: str = "./results"

    # --------------------------------------------------------
    # Checkpoint save frequency
    # --------------------------------------------------------

    save_frequency: int = 5

    # --------------------------------------------------------
    # Loss weights
    #
    # Dice + BCE
    # --------------------------------------------------------

    dice_weight: float = 0.5

    bce_weight: float = 0.5


# ============================================================
# MASTER CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Master configuration object."""

    dataset: DatasetConfig = field(
        default_factory=DatasetConfig
    )

    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    train: TrainConfig = field(
        default_factory=TrainConfig
    )

    def __post_init__(self):

        # ----------------------------------------------------
        # Create required directories
        # ----------------------------------------------------

        os.makedirs(
            self.train.checkpoint_dir,
            exist_ok=True,
        )

        os.makedirs(
            self.train.log_dir,
            exist_ok=True,
        )

        os.makedirs(
            self.train.results_dir,
            exist_ok=True,
        )


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

def get_default_config() -> Config:
    """
    Create and return the default project configuration.
    """

    return Config()
