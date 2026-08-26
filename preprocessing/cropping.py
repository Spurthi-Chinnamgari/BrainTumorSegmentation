import numpy as np


def get_bounding_box(image):

    brain_mask = image > 0

    x, y, z = np.where(brain_mask)

    return (
        x.min(),
        x.max(),
        y.min(),
        y.max(),
        z.min(),
        z.max()
    )


def crop_volume(image, bbox, padding=5):

    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    xmin = max(0, xmin - padding)
    xmax = min(image.shape[0] - 1, xmax + padding)

    ymin = max(0, ymin - padding)
    ymax = min(image.shape[1] - 1, ymax + padding)

    zmin = max(0, zmin - padding)
    zmax = min(image.shape[2] - 1, zmax + padding)

    return image[
        xmin:xmax + 1,
        ymin:ymax + 1,
        zmin:zmax + 1
    ]


def crop_patient(patient):

    bbox = get_bounding_box(
        patient["modalities"]["t1n"]
    )

    for modality in patient["modalities"]:

        patient["modalities"][modality] = crop_volume(
            patient["modalities"][modality],
            bbox
        )

    patient["segmentation"] = crop_volume(
        patient["segmentation"],
        bbox
    )

    return patient