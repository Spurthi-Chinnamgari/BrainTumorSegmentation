"""
Logging and Metric Tracking Utilities.
Supports Console Logging, TensorBoard Event Logging, and Metric Aggregation.
"""

from typing import Dict, Any, Optional
import os
import logging
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None
    HAS_TENSORBOARD = False


def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Initialize and return a formatted console and file logger.
    """
    logger = logging.getLogger("AttentionUNet")
    logger.setLevel(level)
    logger.handlers = []

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricTracker:
    """
    Computes and stores running averages and current values for training/validation metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class TensorBoardLogger:
    """
    TensorBoard Writer wrapper for logging training losses, validation metrics, and image samples.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
            print("[INFO] TensorBoard is not installed. Tensorboard logging will be skipped.")

    def log_scalar(self, tag: str, scalar_value: float, global_step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int):
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def log_image(self, tag: str, img_tensor: Any, global_step: int):
        if self.writer is not None:
            self.writer.add_image(tag, img_tensor, global_step)

    def log_figure(self, tag: str, figure: Any, global_step: int):
        if self.writer is not None:
            self.writer.add_figure(tag, figure, global_step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
