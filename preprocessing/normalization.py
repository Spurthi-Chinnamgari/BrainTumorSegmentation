import numpy as np


def normalize_volume(image):

    brain_mask = image > 0

    brain_voxels = image[brain_mask]

    mean = np.mean(brain_voxels)
    std = np.std(brain_voxels)

    normalized = image.copy()

    normalized[brain_mask] = (
        brain_voxels - mean
    ) / std

    return normalized


def normalize_patient(patient):

    for modality in patient["modalities"]:

        patient["modalities"][modality] = normalize_volume(
            patient["modalities"][modality]
        )

    return patient