import os
import nibabel as nib
import numpy as np

patient_id = "BraTS-GLI-00000-000"

patient_dir = os.path.join(
    "data", 
    "brats2023",
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    patient_id
)

seg_path = os.path.join(
    patient_dir,
    f"{patient_id}-seg.nii.gz"
)

seg = nib.load(seg_path).get_fdata()
tumour_pixels = []
for i in range(seg.shape[2]):
    current_slice = seg[:, :, i]
    count = np.count_nonzero(current_slice)
    tumour_pixels.append(count)
best_slice = np.argmax(tumour_pixels)

print("Best Slice:", best_slice)
print("Tumour Pixels:", tumour_pixels[best_slice])