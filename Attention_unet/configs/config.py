"""
Configuration module for 3D Attention U-Net
Multimodal Brain Tumor Segmentation (BraTS 2023).

Current baseline:
    - 3D Attention U-Net
    - 4 MRI modalities
    - Processed 64x64x64 patches
    - Exact train/validation/test split files
    - 4 segmentation classes
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
    """Dataset and processed-data configuration."""

    # --------------------------------------------------------
    # Raw BraTS 2023 dataset
    #
    # Used by preprocessing only.
    # --------------------------------------------------------

    raw_data_dir: str = (
        r"C:\Datasets\BraTS2023"
        r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )

    # --------------------------------------------------------
    # Processed dataset
    #
    # Expected structure:
    #
    # data/processed/
    #     train/
    #         patient_id/
    #             images.npy
    #             masks.npy
    #     val/
    #         patient_id/
    #             images.npy
    #             masks.npy
    #     test/
    #         patient_id/
    #             images.npy
    #             masks.npy
    # --------------------------------------------------------

    processed_data_dir: str = (
        "./data/processed"
    )

    # --------------------------------------------------------
    # Exact split files
    #
    # Expected:
    #
    # data/splits/train.txt
    # data/splits/val.txt
    # data/splits/test.txt
    # --------------------------------------------------------

    split_dir: str = (
        "./data/splits"
    )

    train_split_file: str = (
        "./data/splits/train.txt"
    )

    val_split_file: str = (
        "./data/splits/val.txt"
    )

    test_split_file: str = (
        "./data/splits/test.txt"
    )

    # --------------------------------------------------------
    # MRI modalities
    #
    # Channel order MUST remain:
    #
    # 0 = T1n
    # 1 = T1c
    # 2 = T2w
    # 3 = T2f
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
    # 1 = NCR/NET
    # 2 = Edema
    # 3 = Enhancing Tumor
    # --------------------------------------------------------

    num_classes: int = 4

    # --------------------------------------------------------
    # Processed 3D patch size
    #
    # MUST match preprocessing and train.py.
    # --------------------------------------------------------

    patch_size_3d: Tuple[int, int, int] = (
        64,
        64,
        64,
    )

    # --------------------------------------------------------
    # Dataset split information
    #
    # The actual split is controlled by train.txt,
    # val.txt and test.txt.
    #
    # These values are retained only as documentation.
    # --------------------------------------------------------

    expected_train_patients: int = 26

    expected_val_patients: int = 6

    expected_test_patients: int = 5

    seed: int = 42


# ============================================================
# MODEL CONFIGURATION
# ============================================================

@dataclass
class ModelConfig:
    """3D Attention U-Net model configuration."""

    # --------------------------------------------------------
    # Model dimension
    # --------------------------------------------------------

    dimension: str = "3d"

    # --------------------------------------------------------
    # Input / output
    # --------------------------------------------------------

    in_channels: int = 4

    out_channels: int = 4

    # --------------------------------------------------------
    # Encoder feature sizes
    #
    # MUST match AttentionUNet3D used for training.
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
    # Dice + multiclass Cross Entropy
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