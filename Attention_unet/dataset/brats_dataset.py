"""
BraTS 2023 Dataset Loader
-------------------------

Real BraTS 2023 data only.

Supported:
    - BraTSDataset3D
    - BraTSDataset2D
    - create_train_val_test_split
    - verify_patient

NO synthetic data is generated or used.

Expected BraTS patient folder:

BraTS-GLI-xxxxx-xxx/
    BraTS-GLI-xxxxx-xxx-t1n.nii.gz
    BraTS-GLI-xxxxx-xxx-t1c.nii.gz
    BraTS-GLI-xxxxx-xxx-t2w.nii.gz
    BraTS-GLI-xxxxx-xxx-t2f.nii.gz
    BraTS-GLI-xxxxx-xxx-seg.nii.gz
"""

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch

from torch.utils.data import Dataset


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MODALITIES = [
    "t1n",
    "t1c",
    "t2w",
    "t2f",
]

SEGMENTATION_SUFFIX = "-seg.nii.gz"


# ============================================================
# FIND NIFTI FILE
# ============================================================

def find_nifti_file(
    patient_dir: str,
    patient_id: str,
    modality: str,
) -> Optional[str]:
    """
    Find a modality file inside a patient directory.

    Supports:
        .nii.gz
        .nii
    """

    candidates = [
        os.path.join(
            patient_dir,
            f"{patient_id}-{modality}.nii.gz"
        ),

        os.path.join(
            patient_dir,
            f"{patient_id}-{modality}.nii"
        ),
    ]

    for path in candidates:

        if os.path.isfile(path):
            return path

    # --------------------------------------------------------
    # Fallback search
    # --------------------------------------------------------

    try:

        for filename in os.listdir(patient_dir):

            lower = filename.lower()

            if (
                f"-{modality}.nii.gz" in lower
                or f"-{modality}.nii" in lower
            ):

                return os.path.join(
                    patient_dir,
                    filename
                )

    except OSError:
        pass

    return None


# ============================================================
# FIND SEGMENTATION
# ============================================================

def find_segmentation_file(
    patient_dir: str,
    patient_id: str,
) -> Optional[str]:
    """
    Find ground-truth segmentation file.
    """

    candidates = [
        os.path.join(
            patient_dir,
            f"{patient_id}-seg.nii.gz"
        ),

        os.path.join(
            patient_dir,
            f"{patient_id}-seg.nii"
        ),
    ]

    for path in candidates:

        if os.path.isfile(path):
            return path

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    try:

        for filename in os.listdir(patient_dir):

            lower = filename.lower()

            if (
                lower.endswith("-seg.nii.gz")
                or lower.endswith("-seg.nii")
            ):

                return os.path.join(
                    patient_dir,
                    filename
                )

    except OSError:
        pass

    return None


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(
    path: str,
) -> np.ndarray:
    """
    Load NIfTI file as numpy array.

    Returns float32 array.
    """

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"NIfTI file not found:\n{path}"
        )

    nii = nib.load(path)

    data = nii.get_fdata(
        dtype=np.float32
    )

    return np.asarray(
        data,
        dtype=np.float32
    )


# ============================================================
# LOAD PATIENT
# ============================================================

def load_patient(
    patient_dir: str,
    modalities: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one complete BraTS patient.

    Returns:

        image:
            (C, H, W, D)

        mask:
            (H, W, D)

    Modalities:
        t1n
        t1c
        t2w
        t2f
    """

    if modalities is None:
        modalities = DEFAULT_MODALITIES

    patient_dir = os.path.abspath(
        patient_dir
    )

    if not os.path.isdir(patient_dir):

        raise FileNotFoundError(
            f"Patient directory not found:\n"
            f"{patient_dir}"
        )

    patient_id = os.path.basename(
        os.path.normpath(patient_dir)
    )

    # ========================================================
    # LOAD MODALITIES
    # ========================================================

    images = []

    reference_shape = None

    for modality in modalities:

        path = find_nifti_file(
            patient_dir,
            patient_id,
            modality,
        )

        if path is None:

            raise FileNotFoundError(
                "\nMissing modality:\n"
                f"Patient: {patient_id}\n"
                f"Modality: {modality}\n"
                f"Directory: {patient_dir}"
            )

        image = load_nifti(path)

        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        if image.ndim != 3:

            raise ValueError(
                f"Expected 3D MRI for {modality}, "
                f"but got shape {image.shape}\n"
                f"File: {path}"
            )

        if reference_shape is None:

            reference_shape = image.shape

        elif image.shape != reference_shape:

            raise ValueError(
                "\nModality shape mismatch.\n"
                f"Patient: {patient_id}\n"
                f"Expected: {reference_shape}\n"
                f"Got: {image.shape}\n"
                f"Modality: {modality}"
            )

        images.append(image)

    # ========================================================
    # LOAD SEGMENTATION
    # ========================================================

    seg_path = find_segmentation_file(
        patient_dir,
        patient_id,
    )

    if seg_path is None:

        raise FileNotFoundError(
            "\nSegmentation file not found.\n"
            f"Patient: {patient_id}\n"
            f"Expected: {patient_id}-seg.nii.gz\n"
            f"Directory: {patient_dir}"
        )

    mask = load_nifti(
        seg_path
    )

    # --------------------------------------------------------
    # Check mask shape
    # --------------------------------------------------------

    if mask.ndim != 3:

        raise ValueError(
            f"Segmentation must be 3D.\n"
            f"Got shape: {mask.shape}\n"
            f"File: {seg_path}"
        )

    if mask.shape != reference_shape:

        raise ValueError(
            "\nImage/mask shape mismatch.\n"
            f"Patient: {patient_id}\n"
            f"Image: {reference_shape}\n"
            f"Mask: {mask.shape}"
        )

    # ========================================================
    # STACK MODALITIES
    # ========================================================

    image = np.stack(
        images,
        axis=0
    )

    # ========================================================
    # CLEAN MASK
    # ========================================================

    # BraTS labels:
    #
    # 0 = background
    # 1 = NCR / NET
    # 2 = edema
    # 3 = enhancing tumor
    #
    # Convert to integer labels.

    mask = np.rint(mask).astype(
        np.int64
    )

    # --------------------------------------------------------
    # Verify labels
    # --------------------------------------------------------

    unique_labels = np.unique(mask)

    invalid_labels = [
        int(x)
        for x in unique_labels
        if x not in [0, 1, 2, 3]
    ]

    if len(invalid_labels) > 0:

        raise ValueError(
            "\nInvalid BraTS segmentation labels.\n"
            f"Patient: {patient_id}\n"
            f"Labels found: {unique_labels}\n"
            f"Invalid labels: {invalid_labels}"
        )

    return image, mask


# ============================================================
# FIND ALL PATIENT DIRECTORIES
# ============================================================

def find_patient_directories(
    data_dir: str,
) -> List[str]:
    """
    Find all BraTS patient directories.

    Only directories containing a segmentation file
    are considered valid patients.
    """

    data_dir = os.path.abspath(
        data_dir
    )

    if not os.path.isdir(data_dir):

        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{data_dir}"
        )

    patient_dirs = []

    for name in sorted(
        os.listdir(data_dir)
    ):

        path = os.path.join(
            data_dir,
            name
        )

        if not os.path.isdir(path):
            continue

        # ----------------------------------------------------
        # Check whether this is a patient directory
        # ----------------------------------------------------

        seg_file = find_segmentation_file(
            path,
            name,
        )

        if seg_file is not None:

            patient_dirs.append(
                path
            )

    return patient_dirs


# ============================================================
# CHECK PATIENT
# ============================================================

def check_patient(
    patient_dir: str,
    modalities: Optional[List[str]] = None,
) -> bool:
    """
    Check whether a patient contains all required files.
    """

    if modalities is None:
        modalities = DEFAULT_MODALITIES

    patient_id = os.path.basename(
        os.path.normpath(patient_dir)
    )

    # --------------------------------------------------------
    # Check modalities
    # --------------------------------------------------------

    for modality in modalities:

        path = find_nifti_file(
            patient_dir,
            patient_id,
            modality,
        )

        if path is None:
            return False

    # --------------------------------------------------------
    # Check segmentation
    # --------------------------------------------------------

    seg = find_segmentation_file(
        patient_dir,
        patient_id,
    )

    if seg is None:
        return False

    return True


# ============================================================
# CREATE TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_train_val_test_split(
    data_dir: str,
    val_split: float = 0.20,
    test_split: float = 0.10,
    seed: int = 42,
):
    """
    Create reproducible train/validation/test split.

    Example:

        1245 patients

        Training   = 872
        Validation = 249
        Testing    = 124

    Returns:

        train_dirs
        val_dirs
        test_dirs
    """

    patient_dirs = find_patient_directories(
        data_dir
    )

    if len(patient_dirs) == 0:

        raise RuntimeError(
            "\nNo BraTS patients found.\n"
            f"Dataset directory:\n{data_dir}"
        )

    # ========================================================
    # SHUFFLE
    # ========================================================

    rng = random.Random(
        seed
    )

    patient_dirs = patient_dirs.copy()

    rng.shuffle(
        patient_dirs
    )

    # ========================================================
    # NUMBER OF PATIENTS
    # ========================================================

    total = len(
        patient_dirs
    )

    num_test = int(
        total * test_split
    )

    num_val = int(
        total * val_split
    )

    # --------------------------------------------------------
    # Make sure we don't remove too many patients
    # --------------------------------------------------------

    if num_test + num_val >= total:

        raise ValueError(
            "Validation + test split is too large."
        )

    # ========================================================
    # SPLIT
    # ========================================================

    test_dirs = patient_dirs[
        :num_test
    ]

    val_dirs = patient_dirs[
        num_test:
        num_test + num_val
    ]

    train_dirs = patient_dirs[
        num_test + num_val:
    ]

    # ========================================================
    # PRINT
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "BraTS DATASET SPLIT"
    )

    print(
        "=" * 60
    )

    print(
        f"Total patients: {total}"
    )

    print(
        f"Training:       {len(train_dirs)}"
    )

    print(
        f"Validation:     {len(val_dirs)}"
    )

    print(
        f"Testing:        {len(test_dirs)}"
    )

    print(
        "=" * 60
    )

    return (
        train_dirs,
        val_dirs,
        test_dirs,
    )


# ============================================================
# 3D DATASET
# ============================================================

class BraTSDataset3D(Dataset):
    """
    BraTS 2023 3D dataset.

    Training:

        image
            (4, D, H, W)

        mask
            (D, H, W)

    Validation:

        complete volume is returned.

    RandomCrop3D is applied through the transform.
    """

    def __init__(
        self,
        patient_dirs: List[str],
        modalities: Optional[List[str]] = None,
        transform=None,
        is_train: bool = True,
    ):

        self.patient_dirs = list(
            patient_dirs
        )

        self.modalities = (
            modalities
            if modalities is not None
            else DEFAULT_MODALITIES.copy()
        )

        self.transform = transform

        self.is_train = is_train

        if len(self.patient_dirs) == 0:

            raise ValueError(
                "BraTSDataset3D received "
                "zero patient directories."
            )

    def __len__(self):

        return len(
            self.patient_dirs
        )

    def __getitem__(
        self,
        index: int,
    ):

        patient_dir = self.patient_dirs[
            index
        ]

        # ====================================================
        # LOAD PATIENT
        # ====================================================

        image, mask = load_patient(
            patient_dir=patient_dir,
            modalities=self.modalities,
        )

        # ====================================================
        # TRANSFORMS
        # ====================================================

        if self.transform is not None:

            image, mask = self.transform(
                image,
                mask
            )

        # ====================================================
        # MAKE SURE TENSOR
        # ====================================================

        if isinstance(image, np.ndarray):

            image = np.ascontiguousarray(
                image
            )

            image = torch.from_numpy(
                image
            ).float()

        if isinstance(mask, np.ndarray):

            mask = np.ascontiguousarray(
                mask
            )

            mask = torch.from_numpy(
                mask
            ).long()

        # ====================================================
        # RETURN
        # ====================================================

        return {
            "image": image,
            "mask": mask,
            "patient_id": os.path.basename(
                os.path.normpath(
                    patient_dir
                )
            ),
        }


# ============================================================
# 2D DATASET
# ============================================================

class BraTSDataset2D(Dataset):
    """
    BraTS 2023 2D dataset.

    Converts every 3D patient volume into individual slices.

    Image:
        (4, H, W)

    Mask:
        (H, W)
    """

    def __init__(
        self,
        patient_dirs: List[str],
        modalities: Optional[List[str]] = None,
        transform=None,
        slice_axis: int = 2,
    ):

        self.patient_dirs = list(
            patient_dirs
        )

        self.modalities = (
            modalities
            if modalities is not None
            else DEFAULT_MODALITIES.copy()
        )

        self.transform = transform

        self.slice_axis = slice_axis

        if len(self.patient_dirs) == 0:

            raise ValueError(
                "BraTSDataset2D received "
                "zero patient directories."
            )

        # ====================================================
        # CREATE SLICE INDEX
        # ====================================================

        self.samples = []

        for patient_index, patient_dir in enumerate(
            self.patient_dirs
        ):

            patient_id = os.path.basename(
                os.path.normpath(
                    patient_dir
                )
            )

            # ------------------------------------------------
            # Find one modality to determine depth
            # ------------------------------------------------

            t1n_path = find_nifti_file(
                patient_dir,
                patient_id,
                self.modalities[0],
            )

            if t1n_path is None:

                continue

            volume = nib.load(
                t1n_path
            )

            shape = volume.shape

            if len(shape) != 3:

                continue

            num_slices = shape[
                self.slice_axis
            ]

            for slice_index in range(
                num_slices
            ):

                self.samples.append(
                    (
                        patient_index,
                        slice_index,
                    )
                )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ):

        patient_index, slice_index = (
            self.samples[index]
        )

        patient_dir = self.patient_dirs[
            patient_index
        ]

        image_3d, mask_3d = load_patient(
            patient_dir,
            self.modalities,
        )

        # ====================================================
        # EXTRACT SLICE
        # ====================================================

        if self.slice_axis == 0:

            image = image_3d[
                :,
                slice_index,
                :,
                :
            ]

            mask = mask_3d[
                slice_index,
                :,
                :
            ]

        elif self.slice_axis == 1:

            image = image_3d[
                :,
                :,
                slice_index,
                :
            ]

            mask = mask_3d[
                :,
                slice_index,
                :
            ]

        else:

            image = image_3d[
                :,
                :,
                :,
                slice_index
            ]

            mask = mask_3d[
                :,
                :,
                slice_index
            ]

        # ====================================================
        # TRANSFORMS
        # ====================================================

        if self.transform is not None:

            image, mask = self.transform(
                image,
                mask
            )

        # ====================================================
        # TO TENSOR
        # ====================================================

        if isinstance(image, np.ndarray):

            image = np.ascontiguousarray(
                image
            )

            image = torch.from_numpy(
                image
            ).float()

        if isinstance(mask, np.ndarray):

            mask = np.ascontiguousarray(
                mask
            )

            mask = torch.from_numpy(
                mask
            ).long()

        # ====================================================
        # RETURN
        # ====================================================

        patient_id = os.path.basename(
            os.path.normpath(
                patient_dir
            )
        )

        return {
            "image": image,
            "mask": mask,
            "patient_id": patient_id,
            "slice_index": slice_index,
        }


# ============================================================
# VERIFY PATIENT
# ============================================================

def verify_patient(
    patient_dir: str,
    modalities: Optional[List[str]] = None,
):
    """
    Load and verify one BraTS patient.

    Prints:
        file sizes
        image shape
        mask shape
        mask labels
    """

    if modalities is None:
        modalities = DEFAULT_MODALITIES

    patient_dir = os.path.abspath(
        patient_dir
    )

    patient_id = os.path.basename(
        os.path.normpath(
            patient_dir
        )
    )

    print()

    print(
        "=" * 70
    )

    print(
        f"VERIFYING PATIENT: {patient_id}"
    )

    print(
        "=" * 70
    )

    print(
        "Directory:"
    )

    print(
        patient_dir
    )

    print()

    # ========================================================
    # FILE SIZES
    # ========================================================

    for modality in modalities:

        path = find_nifti_file(
            patient_dir,
            patient_id,
            modality,
        )

        if path is None:

            print(
                f"{modality}: MISSING"
            )

            continue

        size_kb = (
            os.path.getsize(path)
            / 1024
        )

        print(
            f"{modality}: "
            f"{size_kb:.2f} KB"
        )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    seg_path = find_segmentation_file(
        patient_dir,
        patient_id,
    )

    if seg_path is None:

        print(
            "seg: MISSING"
        )

        return False

    seg_size_kb = (
        os.path.getsize(seg_path)
        / 1024
    )

    print(
        f"seg: {seg_size_kb:.2f} KB"
    )

    print()

    # ========================================================
    # LOAD
    # ========================================================

    try:

        image, mask = load_patient(
            patient_dir,
            modalities,
        )

    except Exception as e:

        print(
            "ERROR:"
        )

        print(
            str(e)
        )

        return False

    # ========================================================
    # PRINT SHAPES
    # ========================================================

    print(
        "IMAGE SHAPE:",
        image.shape,
    )

    print(
        "MASK SHAPE:",
        mask.shape,
    )

    print(
        "MASK LABELS:",
        np.unique(mask),
    )

    # ========================================================
    # VERIFY LABELS
    # ========================================================

    labels = np.unique(
        mask
    )

    invalid = [
        int(x)
        for x in labels
        if x not in [0, 1, 2, 3]
    ]

    if len(invalid) > 0:

        print()

        print(
            "WARNING: "
            f"Unexpected segmentation labels: "
            f"{set(invalid)}"
        )

        return False

    print()

    print(
        "Everything looks good."
    )

    print(
        "=" * 70
    )

    return True


# ============================================================
# CHECK ALL PATIENTS
# ============================================================

def verify_dataset(
    data_dir: str,
    modalities: Optional[List[str]] = None,
    max_patients: Optional[int] = None,
):
    """
    Verify multiple BraTS patients.
    """

    if modalities is None:
        modalities = DEFAULT_MODALITIES

    patient_dirs = find_patient_directories(
        data_dir
    )

    print()

    print(
        "=" * 70
    )

    print(
        "BRAts DATASET VERIFICATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Patients found: {len(patient_dirs)}"
    )

    print(
        "=" * 70
    )

    if max_patients is not None:

        patient_dirs = patient_dirs[
            :max_patients
        ]

    valid_count = 0

    for patient_dir in patient_dirs:

        if verify_patient(
            patient_dir,
            modalities,
        ):

            valid_count += 1

    print()

    print(
        "=" * 70
    )

    print(
        f"Valid BraTS patients: "
        f"{valid_count}"
    )

    print(
        "=" * 70
    )

    return valid_count


# ============================================================
# NO SYNTHETIC DATA
# ============================================================

def generate_synthetic_brats_data(*args, **kwargs):
    """
    Kept only for backward compatibility.

    This project intentionally DOES NOT generate synthetic
    BraTS data.

    If old code tries to call this function, fail clearly.
    """

    raise RuntimeError(
        "\nSynthetic BraTS data generation is disabled.\n"
        "This project uses REAL BraTS 2023 data only.\n"
        "Remove the call to generate_synthetic_brats_data()."
    )


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Verify REAL BraTS 2023 dataset"
        )
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to BraTS dataset directory",
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=5,
        help="Maximum number of patients to verify",
    )

    args = parser.parse_args()

    # ========================================================
    # FIND PATIENTS
    # ========================================================

    print()

    print(
        "Searching for BraTS patients..."
    )

    patient_dirs = find_patient_directories(
        args.data_dir
    )

    print(
        f"Patients found: "
        f"{len(patient_dirs)}"
    )

    # ========================================================
    # PRINT FIRST PATIENTS
    # ========================================================

    for patient_dir in patient_dirs[
        :5
    ]:

        print(
            patient_dir
        )

    # ========================================================
    # VERIFY
    # ========================================================

    verify_dataset(
        data_dir=args.data_dir,
        modalities=DEFAULT_MODALITIES,
        max_patients=args.max_patients,
    )