import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Configuration
# ============================================================

NUM_MODALITIES = 4
NUM_CLASSES = 4
PATCH_SIZE = (64, 64, 64)

MODALITY_NAMES = ["t1n", "t1c", "t2w", "t2f"]


# ============================================================
# Utility functions
# ============================================================

def read_split_file(split_file: str) -> List[str]:
    """
    Read exact patient IDs from a split file.

    Example:
        data/splits/train.txt
        data/splits/val.txt
        data/splits/test.txt
    """

    if not os.path.isfile(split_file):
        raise FileNotFoundError(
            f"Split file not found:\n{split_file}"
        )

    patient_ids = []

    with open(split_file, "r") as f:
        for line in f:
            patient_id = line.strip()

            if patient_id:
                patient_ids.append(patient_id)

    if len(patient_ids) == 0:
        raise RuntimeError(
            f"No patient IDs found in split file:\n{split_file}"
        )

    return patient_ids


def verify_processed_patient(
    processed_root: str,
    split: str,
    patient_id: str,
    expected_patch_size: Tuple[int, int, int] = PATCH_SIZE,
) -> bool:
    """
    Verify that a processed patient contains:
        images.npy
        masks.npy

    Expected:
        images -> (N, 4, 64, 64, 64)
        masks  -> (N, 64, 64, 64)
    """

    patient_dir = os.path.join(
        processed_root,
        split,
        patient_id
    )

    image_file = os.path.join(patient_dir, "images.npy")
    mask_file = os.path.join(patient_dir, "masks.npy")

    if not os.path.isfile(image_file):
        raise FileNotFoundError(
            f"Missing images.npy for patient {patient_id}:\n"
            f"{image_file}"
        )

    if not os.path.isfile(mask_file):
        raise FileNotFoundError(
            f"Missing masks.npy for patient {patient_id}:\n"
            f"{mask_file}"
        )

    images = np.load(image_file, mmap_mode="r")
    masks = np.load(mask_file, mmap_mode="r")

    expected_image_shape = (
        images.shape[0],
        NUM_MODALITIES,
        *expected_patch_size
    )

    expected_mask_shape = (
        masks.shape[0],
        *expected_patch_size
    )

    if images.ndim != 5:
        raise ValueError(
            f"Invalid image shape for {patient_id}: "
            f"{images.shape}\n"
            f"Expected 5 dimensions: "
            f"(N, 4, 64, 64, 64)"
        )

    if masks.ndim != 4:
        raise ValueError(
            f"Invalid mask shape for {patient_id}: "
            f"{masks.shape}\n"
            f"Expected 4 dimensions: "
            f"(N, 64, 64, 64)"
        )

    if images.shape[1:] != (
        NUM_MODALITIES,
        *expected_patch_size
    ):
        raise ValueError(
            f"Invalid image shape for {patient_id}: "
            f"{images.shape}\n"
            f"Expected: "
            f"(N, {NUM_MODALITIES}, "
            f"{expected_patch_size[0]}, "
            f"{expected_patch_size[1]}, "
            f"{expected_patch_size[2]})"
        )

    if masks.shape[1:] != expected_patch_size:
        raise ValueError(
            f"Invalid mask shape for {patient_id}: "
            f"{masks.shape}\n"
            f"Expected: "
            f"(N, "
            f"{expected_patch_size[0]}, "
            f"{expected_patch_size[1]}, "
            f"{expected_patch_size[2]})"
        )

    if images.shape[0] != masks.shape[0]:
        raise ValueError(
            f"Patch count mismatch for {patient_id}: "
            f"images={images.shape[0]}, "
            f"masks={masks.shape[0]}"
        )

    return True


# ============================================================
# Main 3D Dataset
# ============================================================

class BraTSDataset3D(Dataset):
    """
    Dataset for the VERIFIED preprocessing output.

    Expected folder structure:

    data/
    ├── processed/
    │   ├── train/
    │   │   ├── BraTS-GLI-xxxxx/
    │   │   │   ├── images.npy
    │   │   │   └── masks.npy
    │   │   └── ...
    │   │
    │   ├── val/
    │   │   └── ...
    │   │
    │   └── test/
    │       └── ...
    │
    └── splits/
        ├── train.txt
        ├── val.txt
        └── test.txt


    Each images.npy:

        (N, 4, 64, 64, 64)

    Each masks.npy:

        (N, 64, 64, 64)
    """

    def __init__(
        self,
        processed_root: str,
        split: str,
        split_file: Optional[str] = None,
        patch_size: Tuple[int, int, int] = PATCH_SIZE,
    ):
        super().__init__()

        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Use 'train', 'val', or 'test'."
            )

        self.processed_root = os.path.abspath(processed_root)
        self.split = split
        self.patch_size = patch_size

        # ----------------------------------------------------
        # Split file
        # ----------------------------------------------------

        if split_file is None:
            raise ValueError(
                "split_file must be provided.\n"
                "Use the exact train.txt / val.txt / test.txt "
                "created by preprocessing."
            )

        self.split_file = os.path.abspath(split_file)

        self.patient_ids = read_split_file(
            self.split_file
        )

        # ----------------------------------------------------
        # Collect all patches
        # ----------------------------------------------------

        self.samples = []

        print("\n" + "=" * 70)
        print(f"Loading processed {split.upper()} dataset")
        print("=" * 70)

        print(f"Processed root : {self.processed_root}")
        print(f"Split file     : {self.split_file}")
        print(f"Patients       : {len(self.patient_ids)}")

        for patient_id in self.patient_ids:

            verify_processed_patient(
                processed_root=self.processed_root,
                split=split,
                patient_id=patient_id,
                expected_patch_size=patch_size,
            )

            patient_dir = os.path.join(
                self.processed_root,
                split,
                patient_id
            )

            image_file = os.path.join(
                patient_dir,
                "images.npy"
            )

            mask_file = os.path.join(
                patient_dir,
                "masks.npy"
            )

            images = np.load(
                image_file,
                mmap_mode="r"
            )

            patch_count = images.shape[0]

            print(
                f"{patient_id}: "
                f"{patch_count} patches"
            )

            for patch_index in range(patch_count):

                self.samples.append(
                    (
                        image_file,
                        mask_file,
                        patch_index,
                        patient_id,
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No processed patches found for {split} split."
            )

        print("-" * 70)
        print(
            f"{split.upper()} patients : "
            f"{len(self.patient_ids)}"
        )
        print(
            f"{split.upper()} patches  : "
            f"{len(self.samples)}"
        )
        print("=" * 70 + "\n")

    # --------------------------------------------------------
    # Dataset length
    # --------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------------
    # Get one patch
    # --------------------------------------------------------

    def __getitem__(self, index):

        image_file, mask_file, patch_index, patient_id = \
            self.samples[index]

        # Load only the required patch
        images = np.load(
            image_file,
            mmap_mode="r"
        )

        masks = np.load(
            mask_file,
            mmap_mode="r"
        )

        image = np.asarray(
            images[patch_index],
            dtype=np.float32
        )

        mask = np.asarray(
            masks[patch_index],
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        expected_image_shape = (
            NUM_MODALITIES,
            *self.patch_size
        )

        if image.shape != expected_image_shape:
            raise ValueError(
                f"Invalid image shape for "
                f"{patient_id}, patch {patch_index}: "
                f"{image.shape}\n"
                f"Expected: {expected_image_shape}"
            )

        if mask.shape != self.patch_size:
            raise ValueError(
                f"Invalid mask shape for "
                f"{patient_id}, patch {patch_index}: "
                f"{mask.shape}\n"
                f"Expected: {self.patch_size}"
            )

        # Labels must be 0,1,2,3
        unique_labels = np.unique(mask)

        if not np.all(
            np.isin(
                unique_labels,
                [0, 1, 2, 3]
            )
        ):
            raise ValueError(
                f"Invalid labels in "
                f"{patient_id}, patch {patch_index}: "
                f"{unique_labels}"
            )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.copy()
        )

        mask = torch.from_numpy(
            mask.copy()
        )

        return {
            "image": image,
            "mask": mask,
            "patient_id": patient_id,
            "patch_index": patch_index,
        }


# ============================================================
# Compatibility function
# ============================================================

def create_train_val_test_split(
    processed_root: str,
    split_root: str,
):
    """
    DO NOT randomly split patients here.

    The preprocessing pipeline already created the exact
    train / val / test split.

    This function only reads those existing split files.
    """

    train_file = os.path.join(
        split_root,
        "train.txt"
    )

    val_file = os.path.join(
        split_root,
        "val.txt"
    )

    test_file = os.path.join(
        split_root,
        "test.txt"
    )

    train_ids = read_split_file(train_file)
    val_ids = read_split_file(val_file)
    test_ids = read_split_file(test_file)

    print("\nExact preprocessing split:")
    print(f"Train : {len(train_ids)} patients")
    print(f"Val   : {len(val_ids)} patients")
    print(f"Test  : {len(test_ids)} patients")

    return train_ids, val_ids, test_ids


# ============================================================
# Verification helper
# ============================================================

def verify_patient(
    processed_root: str,
    split: str,
    patient_id: str,
):
    """
    Verify one processed patient.
    """

    return verify_processed_patient(
        processed_root=processed_root,
        split=split,
        patient_id=patient_id,
        expected_patch_size=PATCH_SIZE,
    )