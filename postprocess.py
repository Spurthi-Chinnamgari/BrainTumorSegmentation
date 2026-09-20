import os
import numpy as np
import nibabel as nib
from scipy import ndimage


INPUT_MASK = os.path.join("input", "prediction_mask.nii.gz")
OUTPUT_DIR = "postprocessed"
OUTPUT_MASK = os.path.join(
    OUTPUT_DIR,
    "cleaned_prediction_mask.nii.gz"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("3D BRAIN TUMOR POST-PROCESSING")
print("=" * 60)


# ---------------------------------------------------------
# 1. LOAD PREDICTION MASK
# ---------------------------------------------------------

print("\nLoading prediction mask...")
print("Input:", INPUT_MASK)

nii = nib.load(INPUT_MASK)

prediction = nii.get_fdata()

print("Original shape:", prediction.shape)

prediction = np.rint(prediction).astype(np.uint8)

print("\nUnique labels before processing:")
print(np.unique(prediction))


# ---------------------------------------------------------
# 2. DEFINE SEGMENTATION LABELS
# ---------------------------------------------------------

BACKGROUND = 0
NCR_NET = 1
EDEMA = 2
ENHANCING_TUMOR = 3


# ---------------------------------------------------------
# 3. REMOVE SMALL 3D CONNECTED COMPONENTS
# ---------------------------------------------------------

def remove_small_components(mask, min_size=100):

    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)

    labels = [
        NCR_NET,
        EDEMA,
        ENHANCING_TUMOR
    ]

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=2
    )

    for label in labels:

        print("\nProcessing label:", label)

        binary_mask = (mask == label)

        original_voxels = np.sum(binary_mask)

        print("Original voxels:", original_voxels)

        if original_voxels == 0:
            print("No voxels found for this label.")
            continue

        connected_components, number_of_components = ndimage.label(
            binary_mask,
            structure=structure
        )

        print(
            "Connected components:",
            number_of_components
        )

        component_sizes = np.bincount(
            connected_components.ravel()
        )

        valid_components = np.where(
            component_sizes >= min_size
        )[0]

        valid_components = valid_components[
            valid_components != 0
        ]

        cleaned_label = np.isin(
            connected_components,
            valid_components
        )

        cleaned_voxels = np.sum(cleaned_label)

        print(
            "Voxels after removing tiny components:",
            cleaned_voxels
        )

        # Conservative morphological smoothing
        cleaned_label = ndimage.binary_closing(
            cleaned_label,
            structure=structure,
            iterations=1
        )

        cleaned_mask[cleaned_label] = label

    return cleaned_mask


# ---------------------------------------------------------
# 4. RUN POST-PROCESSING ON COMPLETE 3D VOLUME
# ---------------------------------------------------------

cleaned_mask = remove_small_components(
    prediction,
    min_size=100
)


# ---------------------------------------------------------
# 5. DISPLAY FINAL RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("POST-PROCESSING COMPLETED")
print("=" * 60)

print("\nFinal mask shape:")
print(cleaned_mask.shape)

print("\nFinal labels:")

for label in [0, 1, 2, 3]:

    voxel_count = np.sum(
        cleaned_mask == label
    )

    if label == 0:
        name = "Background"

    elif label == 1:
        name = "NCR / NET"

    elif label == 2:
        name = "Edema"

    else:
        name = "Enhancing Tumor"

    print(
        f"Class {label} ({name}): "
        f"{voxel_count} voxels"
    )


# ---------------------------------------------------------
# 6. SAVE CLEANED 3D MASK
# ---------------------------------------------------------

cleaned_nii = nib.Nifti1Image(
    cleaned_mask,
    nii.affine,
    nii.header
)

nib.save(
    cleaned_nii,
    OUTPUT_MASK
)

print("\nSaved cleaned mask to:")
print(OUTPUT_MASK)

print("\nPost-processing was performed on the")
print("COMPLETE 3D volume.")

print("=" * 60)