"""
Medical Image Preprocessing and Augmentation Transforms
for 2D and 3D BraTS Brain Tumor Segmentation.
"""

from typing import Tuple, Optional, List, Any

import numpy as np
import torch


# ============================================================
# Z-SCORE NORMALIZATION
# ============================================================

class ZScoreNormalize:
    """
    Normalize each MRI modality independently using non-zero
    brain voxels.

    Input:
        2D -> (C, H, W)
        3D -> (C, D, H, W)
    """

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        image = image.astype(np.float32)

        normalized = np.zeros_like(image, dtype=np.float32)

        for c in range(image.shape[0]):

            channel = image[c]

            # Brain/non-background voxels
            nonzero = channel != 0

            if np.any(nonzero):

                values = channel[nonzero]

                mean = values.mean()
                std = values.std()

                if std > 1e-8:
                    normalized[c] = np.where(
                        nonzero,
                        (channel - mean) / std,
                        0.0
                    )
                else:
                    normalized[c] = np.where(
                        nonzero,
                        channel - mean,
                        0.0
                    )

            else:
                normalized[c] = channel

        return normalized, mask


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

class MinMaxNormalize:

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        image = image.astype(np.float32)

        normalized = np.zeros_like(image, dtype=np.float32)

        for c in range(image.shape[0]):

            channel = image[c]

            c_min = channel.min()
            c_max = channel.max()

            if c_max - c_min > 1e-8:
                normalized[c] = (
                    channel - c_min
                ) / (c_max - c_min)

            else:
                normalized[c] = channel

        return normalized, mask


# ============================================================
# RANDOM FLIP - 3D
# ============================================================

class RandomFlip3D:
    """
    Randomly flip 3D MRI volume and mask.

    Image shape:
        (C, D, H, W)

    Mask shape:
        (D, H, W)
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        # Depth
        if np.random.rand() < self.prob:
            image = np.flip(image, axis=1)

            if mask is not None:
                mask = np.flip(mask, axis=0)

        # Height
        if np.random.rand() < self.prob:
            image = np.flip(image, axis=2)

            if mask is not None:
                mask = np.flip(mask, axis=1)

        # Width
        if np.random.rand() < self.prob:
            image = np.flip(image, axis=3)

            if mask is not None:
                mask = np.flip(mask, axis=2)

        image = np.ascontiguousarray(image)

        if mask is not None:
            mask = np.ascontiguousarray(mask)

        return image, mask


# ============================================================
# RANDOM FLIP - 2D
# ============================================================

class RandomFlip2D:
    """
    Randomly flip 2D MRI image and mask.

    Image shape:
        (C, H, W)

    Mask shape:
        (H, W)
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        # Height
        if np.random.rand() < self.prob:
            image = np.flip(image, axis=1)

            if mask is not None:
                mask = np.flip(mask, axis=0)

        # Width
        if np.random.rand() < self.prob:
            image = np.flip(image, axis=2)

            if mask is not None:
                mask = np.flip(mask, axis=1)

        image = np.ascontiguousarray(image)

        if mask is not None:
            mask = np.ascontiguousarray(mask)

        return image, mask


# ============================================================
# RANDOM CROP - 3D
# ============================================================

class RandomCrop3D:
    """
    Randomly crop a 3D patch from the MRI volume.

    Image:
        (C, D, H, W)

    Mask:
        (D, H, W)

    patch_size:
        (D, H, W)
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int]
    ):
        self.patch_size = patch_size

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        _, d, h, w = image.shape

        pd, ph, pw = self.patch_size

        # ----------------------------------------------------
        # Padding if volume is smaller than requested patch
        # ----------------------------------------------------

        pad_d = max(0, pd - d)
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:

            image = np.pad(
                image,
                (
                    (0, 0),
                    (0, pad_d),
                    (0, pad_h),
                    (0, pad_w)
                ),
                mode="constant",
                constant_values=0
            )

            if mask is not None:

                mask = np.pad(
                    mask,
                    (
                        (0, pad_d),
                        (0, pad_h),
                        (0, pad_w)
                    ),
                    mode="constant",
                    constant_values=0
                )

            _, d, h, w = image.shape

        # ----------------------------------------------------
        # Random crop position
        # ----------------------------------------------------

        start_d = np.random.randint(
            0,
            d - pd + 1
        )

        start_h = np.random.randint(
            0,
            h - ph + 1
        )

        start_w = np.random.randint(
            0,
            w - pw + 1
        )

        # ----------------------------------------------------
        # Crop image
        # ----------------------------------------------------

        cropped_image = image[
            :,
            start_d:start_d + pd,
            start_h:start_h + ph,
            start_w:start_w + pw
        ]

        # ----------------------------------------------------
        # Crop mask
        # ----------------------------------------------------

        cropped_mask = None

        if mask is not None:

            cropped_mask = mask[
                start_d:start_d + pd,
                start_h:start_h + ph,
                start_w:start_w + pw
            ]

        cropped_image = np.ascontiguousarray(
            cropped_image
        )

        if cropped_mask is not None:
            cropped_mask = np.ascontiguousarray(
                cropped_mask
            )

        return cropped_image, cropped_mask


# ============================================================
# RANDOM CROP - 2D
# ============================================================

class RandomCrop2D:
    """
    Randomly crop a 2D image and mask.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int]
    ):
        self.crop_size = crop_size

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        _, h, w = image.shape

        ph, pw = self.crop_size

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)

        if pad_h > 0 or pad_w > 0:

            image = np.pad(
                image,
                (
                    (0, 0),
                    (0, pad_h),
                    (0, pad_w)
                ),
                mode="constant",
                constant_values=0
            )

            if mask is not None:

                mask = np.pad(
                    mask,
                    (
                        (0, pad_h),
                        (0, pad_w)
                    ),
                    mode="constant",
                    constant_values=0
                )

            _, h, w = image.shape

        # ----------------------------------------------------
        # Random crop location
        # ----------------------------------------------------

        start_h = np.random.randint(
            0,
            h - ph + 1
        )

        start_w = np.random.randint(
            0,
            w - pw + 1
        )

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        cropped_image = image[
            :,
            start_h:start_h + ph,
            start_w:start_w + pw
        ]

        cropped_mask = None

        if mask is not None:

            cropped_mask = mask[
                start_h:start_h + ph,
                start_w:start_w + pw
            ]

        cropped_image = np.ascontiguousarray(
            cropped_image
        )

        if cropped_mask is not None:
            cropped_mask = np.ascontiguousarray(
                cropped_mask
            )

        return cropped_image, cropped_mask


# ============================================================
# RANDOM ROTATION - 3D
# ============================================================

class RandomRotation3D:
    """
    Random 90-degree rotation for 3D MRI volumes.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        if np.random.rand() < self.prob:

            k = np.random.randint(1, 4)

            planes = [
                (1, 2),
                (1, 3),
                (2, 3)
            ]

            plane = planes[
                np.random.randint(len(planes))
            ]

            image = np.rot90(
                image,
                k=k,
                axes=plane
            )

            if mask is not None:

                mask_plane = (
                    plane[0] - 1,
                    plane[1] - 1
                )

                mask = np.rot90(
                    mask,
                    k=k,
                    axes=mask_plane
                )

        image = np.ascontiguousarray(image)

        if mask is not None:
            mask = np.ascontiguousarray(mask)

        return image, mask


# ============================================================
# INTENSITY JITTER
# ============================================================

class IntensityJitter:

    def __init__(
        self,
        prob: float = 0.3,
        noise_std: float = 0.05,
        gamma_range: Tuple[float, float] = (0.7, 1.5)
    ):
        self.prob = prob
        self.noise_std = noise_std
        self.gamma_range = gamma_range

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        image = image.astype(np.float32)

        # Gaussian noise
        if np.random.rand() < self.prob:

            noise = np.random.normal(
                0,
                self.noise_std,
                size=image.shape
            ).astype(np.float32)

            image = image + noise

        # Gamma augmentation
        if np.random.rand() < self.prob:

            gamma = np.random.uniform(
                self.gamma_range[0],
                self.gamma_range[1]
            )

            min_val = image.min()

            shifted = image - min_val

            max_val = shifted.max()

            if max_val > 1e-8:

                image = (
                    np.power(
                        np.clip(
                            shifted / max_val,
                            0,
                            1
                        ),
                        gamma
                    )
                    * max_val
                    + min_val
                )

        return image.astype(np.float32), mask


# ============================================================
# TO TENSOR
# ============================================================

class ToTensor:

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        image = np.ascontiguousarray(image)

        image_tensor = torch.from_numpy(
            image
        ).float()

        mask_tensor = None

        if mask is not None:

            mask = np.ascontiguousarray(mask)

            mask_tensor = torch.from_numpy(
                mask
            ).long()

        return image_tensor, mask_tensor


# ============================================================
# COMPOSE TRANSFORMS
# ============================================================

class ComposeTransforms:
    """
    Apply transforms sequentially.
    """

    def __init__(
        self,
        transforms: List[Any]
    ):
        self.transforms = transforms

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[Any, Optional[Any]]:

        for transform in self.transforms:

            image, mask = transform(
                image,
                mask
            )

        return image, mask