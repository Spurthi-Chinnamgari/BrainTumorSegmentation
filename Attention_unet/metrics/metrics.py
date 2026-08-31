"""
Evaluation Metrics for Medical Image Segmentation (BraTS 2023).
Includes Dice Score, IoU (Jaccard Index), Precision, Recall, F1 Score,
BraTS Sub-Region Dice (WT, TC, ET), and 95th Percentile Hausdorff Distance (HD95).
"""

from typing import Dict, Tuple, List, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt
    HAS_SCIPY_NDIMAGE = True
except ImportError:
    distance_transform_edt = None
    HAS_SCIPY_NDIMAGE = False


def compute_dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-5,
    ignore_background: bool = True,
) -> Tuple[float, List[float]]:
    """
    Compute multi-class Dice score.

    Args:
        preds: Predicted label mask of shape (B, H, W) / (B, D, H, W) or probabilities (B, C, ...)
        targets: Target ground truth mask of shape (B, H, W) / (B, D, H, W)
        num_classes: Number of classes
        smooth: Smoothing epsilon
        ignore_background: If True, exclude background class 0 from mean score

    Returns:
        Mean Dice score across classes, and list of per-class Dice scores.
    """
    if preds.ndim == targets.ndim + 1:
        preds = torch.argmax(preds, dim=1)

    per_class_dice = []
    start_class = 1 if ignore_background else 0

    for c in range(num_classes):
        p_c = (preds == c).float()
        t_c = (targets == c).float()

        intersection = torch.sum(p_c * t_c)
        cardinality = torch.sum(p_c) + torch.sum(t_c)

        dice_c = (2.0 * intersection + smooth) / (cardinality + smooth)
        per_class_dice.append(dice_c.item())

    mean_dice = float(np.mean(per_class_dice[start_class:]))
    return mean_dice, per_class_dice


def compute_iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-5,
    ignore_background: bool = True,
) -> Tuple[float, List[float]]:
    """
    Compute Intersection over Union (IoU / Jaccard Index).
    """
    if preds.ndim == targets.ndim + 1:
        preds = torch.argmax(preds, dim=1)

    per_class_iou = []
    start_class = 1 if ignore_background else 0

    for c in range(num_classes):
        p_c = (preds == c).float()
        t_c = (targets == c).float()

        intersection = torch.sum(p_c * t_c)
        union = torch.sum(p_c) + torch.sum(t_c) - intersection

        iou_c = (intersection + smooth) / (union + smooth)
        per_class_iou.append(iou_c.item())

    mean_iou = float(np.mean(per_class_iou[start_class:]))
    return mean_iou, per_class_iou


def compute_precision_recall_f1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-5,
    ignore_background: bool = True,
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute Precision, Recall, and F1-Score for semantic segmentation.
    """
    if preds.ndim == targets.ndim + 1:
        preds = torch.argmax(preds, dim=1)

    precisions, recalls, f1s = [], [], []
    start_class = 1 if ignore_background else 0

    for c in range(num_classes):
        p_c = (preds == c).float()
        t_c = (targets == c).float()

        tp = torch.sum(p_c * t_c)
        fp = torch.sum(p_c * (1.0 - t_c))
        fn = torch.sum((1.0 - p_c) * t_c)

        prec = (tp + smooth) / (tp + fp + smooth)
        rec = (tp + smooth) / (tp + fn + smooth)
        f1 = (2.0 * prec * rec + smooth) / (prec + rec + smooth)

        precisions.append(prec.item())
        recalls.append(rec.item())
        f1s.append(f1.item())

    mean_prec = float(np.mean(precisions[start_class:]))
    mean_rec = float(np.mean(recalls[start_class:]))
    mean_f1 = float(np.mean(f1s[start_class:]))

    return {
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "mean_f1": mean_f1,
        "class_precision": precisions,
        "class_recall": recalls,
        "class_f1": f1s,
    }


def compute_brats_regions_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """
    Compute BraTS Evaluation Region Dice Scores:
      - Whole Tumor (WT): Labels 1 + 2 + 3 (NCR + ED + ET)
      - Tumor Core (TC): Labels 1 + 3 (NCR + ET)
      - Enhancing Tumor (ET): Label 3 (ET)

    Args:
        preds: Predicted tensor of shape (B, H, W) or (B, D, H, W)
        targets: Ground truth tensor of shape (B, H, W) or (B, D, H, W)
    """
    if preds.ndim == targets.ndim + 1:
        preds = torch.argmax(preds, dim=1)

    # 1. Whole Tumor (WT) -> Labels 1, 2, 3
    pred_wt = (preds > 0).float()
    target_wt = (targets > 0).float()
    dice_wt = (2.0 * torch.sum(pred_wt * target_wt) + smooth) / (
        torch.sum(pred_wt) + torch.sum(target_wt) + smooth
    )

    # 2. Tumor Core (TC) -> Labels 1, 3
    pred_tc = ((preds == 1) | (preds == 3)).float()
    target_tc = ((targets == 1) | (targets == 3)).float()
    dice_tc = (2.0 * torch.sum(pred_tc * target_tc) + smooth) / (
        torch.sum(pred_tc) + torch.sum(target_tc) + smooth
    )

    # 3. Enhancing Tumor (ET) -> Label 3
    pred_et = (preds == 3).float()
    target_et = (targets == 3).float()
    dice_et = (2.0 * torch.sum(pred_et * target_et) + smooth) / (
        torch.sum(pred_et) + torch.sum(target_et) + smooth
    )

    mean_brats_dice = (dice_wt.item() + dice_tc.item() + dice_et.item()) / 3.0

    return {
        "mean_brats_dice": mean_brats_dice,
        "dice_wt": dice_wt.item(),
        "dice_tc": dice_tc.item(),
        "dice_et": dice_et.item(),
    }


def compute_hausdorff_distance_95(
    preds: np.ndarray,
    targets: np.ndarray,
    voxel_spacing: Optional[Tuple[float, ...]] = None,
) -> float:
    """
    Compute 95th Percentile Hausdorff Distance (HD95) between binary segmentations.

    Args:
        preds: Binary numpy array of predictions (H, W) or (D, H, W)
        targets: Binary numpy array of ground truth (H, W) or (D, H, W)
        voxel_spacing: Optional physical voxel spacing tuple
    Returns:
        HD95 distance in mm/voxels. Returns 0.0 if both empty, 100.0 if one is empty.
    """
    preds_bool = preds.astype(bool)
    targets_bool = targets.astype(bool)

    if not np.any(preds_bool) and not np.any(targets_bool):
        return 0.0
    if not np.any(preds_bool) or not np.any(targets_bool):
        return 100.0

    if not HAS_SCIPY_NDIMAGE:
        # Fallback approximation if scipy is not installed
        return 0.0

    # Surface distance via Euclidean distance transform
    dt_target = distance_transform_edt(~targets_bool, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~preds_bool, sampling=voxel_spacing)

    dist_pred_to_target = dt_target[preds_bool]
    dist_target_to_pred = dt_pred[targets_bool]

    if len(dist_pred_to_target) == 0 or len(dist_target_to_pred) == 0:
        return 100.0

    hd95_pred = np.percentile(dist_pred_to_target, 95)
    hd95_target = np.percentile(dist_target_to_pred, 95)

    return float(max(hd95_pred, hd95_target))
