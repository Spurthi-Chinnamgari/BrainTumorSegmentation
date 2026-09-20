import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

patient_id = "BraTS-GLI-00000-000"
patient_dir = os.path.join(
    "data",
    "brats2023",
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    patient_id
)
modalities = ["t1n", "t1c", "t2w", "t2f", "seg"]

images = {}
for modality in modalities:
    try:
        path = os.path.join(
            patient_dir,
            f"{patient_id}-{modality}.nii.gz"
        )

        print(f"Loading {modality} from {path}")
        image = nib.load(path)
        data = image.get_fdata()
        images[modality] = data
    except Exception as e:
        print(f"Error loading {modality}: {e}")
# print("\nLoaded modalities:")
# print(images.keys())

# slice_number = images['t1n'].shape[2] // 2  # Middle slice for visualization

# find the best slice
tumour_pixels = []

seg = images["seg"]

for i in range(seg.shape[2]):

    current_slice = seg[:, :, i]

    count = np.count_nonzero(current_slice)

    tumour_pixels.append(count)

slice_number = np.argmax(tumour_pixels)

print("Best Slice:", slice_number)
print("Tumour Pixels:", tumour_pixels[slice_number])

print("Displaying Slice:", slice_number)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
titles = ["T1n", "T1c", "T2w", "T2f", "Segmentation"]
for ax, modality, title in zip(axes, modalities, titles):
    if modality == "seg":
        seg_cmap = ListedColormap([
            "black",      # 0 Background
            "red",        # 1
            "yellow",     # 2
            "lime"        # 3
        ])
        ax.imshow(
            images["seg"][:, :, slice_number],
            cmap=seg_cmap,
            vmin=0,
            vmax=3,
            interpolation="nearest"
    )
    else:
        ax.imshow(
            images[modality][:, :, slice_number],
            cmap="gray"
        )
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
