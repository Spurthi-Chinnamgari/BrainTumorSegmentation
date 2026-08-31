"""
Loss Functions for Multimodal Brain Tumor Segmentation.

Designed for multi-class BraTS segmentation with:
    Class 0 = Background
    Class 1 = NCR / NET
    Class 2 = Edema
    Class 3 = Enhancing Tumor

Includes:
    - Soft Dice Loss
    - Multi-Class Cross Entropy Loss
    - Combined Dice + Cross Entropy Loss
    - Focal Loss
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SOFT DICE LOSS
# ============================================================

class SoftDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss.

    The target is expected to contain integer class labels:

        0 -> Background
        1 -> NCR/NET
        2 -> Edema
        3 -> Enhancing Tumor

    logits:
        2D -> (B, C, H, W)
        3D -> (B, C, D, H, W)

    targets:
        2D -> (B, H, W)
        3D -> (B, D, H, W)
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        class_weights: Optional[List[float]] = None,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()

        self.smooth = smooth
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        num_classes = logits.shape[1]

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # ----------------------------------------------------
        # Convert target to one-hot
        # ----------------------------------------------------

        if targets.ndim == logits.ndim - 1:

            # Make sure targets are integer labels
            targets = targets.long()

            targets_one_hot = F.one_hot(
                targets,
                num_classes=num_classes,
            )

            # Example:
            # 3D:
            # (B,D,H,W,C)
            # ->
            # (B,C,D,H,W)

            dims = [0, targets_one_hot.ndim - 1]

            spatial_dims = list(
                range(1, targets_one_hot.ndim - 1)
            )

            targets_one_hot = targets_one_hot.permute(
                0,
                -1,
                *spatial_dims,
            )

            targets_one_hot = targets_one_hot.float()

        else:

            targets_one_hot = targets.float()

        # ----------------------------------------------------
        # Flatten spatial dimensions
        # ----------------------------------------------------

        probs_flat = probs.reshape(
            probs.shape[0],
            num_classes,
            -1,
        )

        targets_flat = targets_one_hot.reshape(
            targets_one_hot.shape[0],
            num_classes,
            -1,
        )

        # ----------------------------------------------------
        # Dice calculation
        # ----------------------------------------------------

        intersection = torch.sum(
            probs_flat * targets_flat,
            dim=2,
        )

        denominator = (
            torch.sum(probs_flat, dim=2)
            + torch.sum(targets_flat, dim=2)
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        dice_loss = 1.0 - dice

        # ----------------------------------------------------
        # Ignore class if requested
        # ----------------------------------------------------

        if self.ignore_index is not None:

            valid_mask = torch.ones(
                num_classes,
                dtype=torch.bool,
                device=logits.device,
            )

            if 0 <= self.ignore_index < num_classes:
                valid_mask[self.ignore_index] = False
                dice_loss = dice_loss[:, valid_mask]

        # ----------------------------------------------------
        # Class weights
        # ----------------------------------------------------

        if self.class_weights is not None:

            weights = torch.tensor(
                self.class_weights,
                dtype=torch.float32,
                device=logits.device,
            )

            if self.ignore_index is not None:
                weights = weights[valid_mask]

            dice_loss = dice_loss * weights.view(1, -1)

        return dice_loss.mean()


# ============================================================
# MULTI-CLASS CROSS ENTROPY LOSS
# ============================================================

class BCEWithLogitsLossWrapper(nn.Module):
    """
    Compatibility wrapper.

    IMPORTANT:
    For BraTS multi-class segmentation, standard
    CrossEntropyLoss is more appropriate than independent
    binary BCE for the mutually-exclusive classes.

    This class is kept with the original name so the rest
    of the project does not break.
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        self.class_weights = class_weights

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        weight = None

        if self.class_weights is not None:

            weight = torch.tensor(
                self.class_weights,
                dtype=torch.float32,
                device=logits.device,
            )

        targets = targets.long()

        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
        )


# ============================================================
# DICE + CROSS ENTROPY LOSS
# ============================================================

class DiceBCELoss(nn.Module):
    """
    Combined Dice + Cross Entropy Loss.

    Despite the historical name DiceBCELoss, the second term
    is now multi-class Cross Entropy, which is appropriate
    for BraTS segmentation.

    Loss =
        dice_weight * DiceLoss
        +
        bce_weight * CrossEntropyLoss
    """

    def __init__(
        self,
        dice_weight: float = 0.7,
        bce_weight: float = 0.3,
        smooth: float = 1e-5,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice_loss = SoftDiceLoss(
            smooth=smooth,
            class_weights=class_weights,
        )

        self.bce_loss = BCEWithLogitsLossWrapper(
            class_weights=class_weights,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        dice = self.dice_loss(
            logits,
            targets,
        )

        ce = self.bce_loss(
            logits,
            targets,
        )

        total_loss = (
            self.dice_weight * dice
            +
            self.bce_weight * ce
        )

        return total_loss


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Useful when classes are highly imbalanced.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        targets = targets.long()

        log_probs = F.log_softmax(
            logits,
            dim=1,
        )

        probs = torch.exp(log_probs)

        # Gather probability corresponding to correct class
        targets_unsqueezed = targets.unsqueeze(1)

        pt = probs.gather(
            1,
            targets_unsqueezed,
        ).squeeze(1)

        log_pt = log_probs.gather(
            1,
            targets_unsqueezed,
        ).squeeze(1)

        focal_weight = (
            self.alpha
            * (1.0 - pt).pow(self.gamma)
        )

        loss = -focal_weight * log_pt

        return loss.mean()