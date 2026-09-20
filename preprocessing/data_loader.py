import os
import nibabel as nib
import numpy as np

from config import MODALITIES, SEGMENTATION


def load_modality(patient_dir: str,
                  patient_id: str,
                  modality: str) -> np.ndarray:

    path = os.path.join(
        patient_dir,
        f"{patient_id}-{modality}.nii.gz"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return nib.load(path).get_fdata()


def load_patient(patient_dir: str,
                 patient_id: str) -> dict:

    patient = {
        "modalities": {},
        "segmentation": None
    }

    # MRI Modalities

    for modality in MODALITIES:

        patient["modalities"][modality] = load_modality(
            patient_dir,
            patient_id,
            modality
        )

    # Segmentation

    patient["segmentation"] = load_modality(
        patient_dir,
        patient_id,
        SEGMENTATION
    )

    return patient

def get_patient_ids(dataset_root: str, percentage=100):
    """
    Get patient IDs from the dataset directory.

    percentage=100 → all patients
    percentage=50  → 50% of patients
    percentage=10  → 10% of patients
    """

    patient_ids = []

    for name in os.listdir(dataset_root):

        patient_dir = os.path.join(
            dataset_root,
            name
        )

        if os.path.isdir(patient_dir):
            patient_ids.append(name)

    patient_ids = sorted(patient_ids)

    total_patients = len(patient_ids)

    if percentage <= 0 or percentage > 100:
        raise ValueError(
            "percentage must be between 1 and 100"
        )

    number_of_patients = max(
        1,
        int(total_patients * percentage / 100)
    )

    return patient_ids[:number_of_patients]