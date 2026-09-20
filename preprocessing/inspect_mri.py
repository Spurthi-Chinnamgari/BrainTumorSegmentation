import nibabel as nib
import numpy as np
import os


patient_id = "BraTS-GLI-00000-000"

patient_dir = os.path.join(
    "data",
    "brats2023",
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    patient_id
)

modalities = ["t1n", "t1c", "t2w", "t2f", "seg"]

for modality in modalities:

    path = os.path.join(
        patient_dir,
        f"{patient_id}-{modality}.nii.gz"
    )

    image = nib.load(path)
    data = image.get_fdata()

    print(f"\n--- {modality.upper()} ---")
    print("Shape:", data.shape)
    print("Min:", np.min(data))
    print("Max:", np.max(data))
    print("Mean:", np.mean(data))
    print("Std:", np.std(data))

    if modality == "seg":
        print("Unique segmentation labels:", np.unique(data))