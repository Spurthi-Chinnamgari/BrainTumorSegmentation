"""
Medical Image Visualization Utilities for BraTS Multimodal MRI and Segmentation Overlays.
Supports color-coded tumor sub-region visualization and side-by-side comparison grids.
"""

from typing import Tuple, List, Optional, Union
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
    TUMOR_COLORMAP = ListedColormap([
        [0.0, 0.0, 0.0, 0.0],  # Class 0: Background (Transparent)
        [0.9, 0.1, 0.1, 0.7],  # Class 1: NCR (Red)
        [0.1, 0.8, 0.2, 0.7],  # Class 2: ED (Green)
        [1.0, 0.8, 0.0, 0.8],  # Class 3: ET (Yellow)
    ])
except (ImportError, ModuleNotFoundError):
    plt = None
    ListedColormap = None
    HAS_MATPLOTLIB = False
    TUMOR_COLORMAP = None


def create_overlay(image_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
    """
    Create RGBA overlay of color-coded segmentation mask on grayscale MRI slice.
    """
    # Normalize grayscale background slice to [0, 1]
    bg_min, bg_max = np.min(image_slice), np.max(image_slice)
    if bg_max - bg_min > 1e-8:
        bg_norm = (image_slice - bg_min) / (bg_max - bg_min)
    else:
        bg_norm = image_slice

    # Convert 2D grayscale to RGB
    rgb = np.stack([bg_norm] * 3, axis=-1)

    # Color mapping
    colors = [
        [0.0, 0.0, 0.0],  # 0: BG
        [0.9, 0.1, 0.1],  # 1: NCR (Red)
        [0.1, 0.8, 0.2],  # 2: ED (Green)
        [1.0, 0.8, 0.0],  # 3: ET (Yellow)
    ]

    overlay_rgb = rgb.copy()
    alpha = 0.5

    for c in range(1, 4):
        c_mask = mask_slice == c
        if np.any(c_mask):
            for ch in range(3):
                overlay_rgb[..., ch] = np.where(
                    c_mask,
                    (1 - alpha) * rgb[..., ch] + alpha * colors[c][ch],
                    overlay_rgb[..., ch],
                )

    return overlay_rgb


def save_prediction_comparison(
    image: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    save_path: str,
    title: str = "BraTS Segmentation Prediction",
    slice_idx: Optional[int] = None,
):
    """
    Save complete comparison grid figure containing:
      Row 1: T1, T1ce, T2, FLAIR modalities
      Row 2: Ground Truth Mask, Predicted Mask, GT Overlay, Prediction Overlay

    Args:
        image: Multi-modal image of shape (4, H, W) or (4, D, H, W)
        target: Ground truth mask of shape (H, W) or (D, H, W)
        pred: Predicted mask of shape (H, W) or (D, H, W)
        save_path: Output filepath to save PNG image figure.
        title: Figure super-title.
        slice_idx: Slice index along depth if 3D volume is provided.
    """
    if image.ndim == 4:
        # 3D volume: shape (4, D, H, W)
        d = image.shape[1]
        if slice_idx is None:
            # Find slice with maximum tumor region in ground truth
            tumor_voxels_per_slice = np.sum(target > 0, axis=(1, 2))
            if np.max(tumor_voxels_per_slice) > 0:
                slice_idx = int(np.argmax(tumor_voxels_per_slice))
            else:
                slice_idx = d // 2

        img_2d = image[:, slice_idx, :, :]
        tgt_2d = target[slice_idx, :, :]
        prd_2d = pred[slice_idx, :, :]
    else:
        # 2D image: shape (4, H, W)
        img_2d = image
        tgt_2d = target
        prd_2d = pred

    if not HAS_MATPLOTLIB:
        print("[INFO] Matplotlib is not installed. Saving visualization arrays to .npz file.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(
            save_path.replace(".png", ".npz"),
            image=img_2d,
            target=tgt_2d,
            prediction=prd_2d,
        )
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"{title} (Slice: {slice_idx if slice_idx is not None else 0})", fontsize=16)

    modalities = ["T1 Native", "T1 Contrast (T1ce)", "T2 Weighted", "T2 FLAIR"]

    # Row 1: 4 MRI Modalities
    for i in range(4):
        axes[0, i].imshow(img_2d[i], cmap="gray")
        axes[0, i].set_title(modalities[i])
        axes[0, i].axis("off")

    # Row 2: Ground Truth, Prediction, GT Overlay, Prediction Overlay
    gt_overlay = create_overlay(img_2d[3], tgt_2d)
    prd_overlay = create_overlay(img_2d[3], prd_2d)

    axes[1, 0].imshow(tgt_2d, cmap=TUMOR_COLORMAP, vmin=0, vmax=3)
    axes[1, 0].set_title("Ground Truth Mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(prd_2d, cmap=TUMOR_COLORMAP, vmin=0, vmax=3)
    axes[1, 1].set_title("Predicted Mask")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(gt_overlay)
    axes[1, 2].set_title("GT Overlay (FLAIR)")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(prd_overlay)
    axes[1, 3].set_title("Prediction Overlay (FLAIR)")
    axes[1, 3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
