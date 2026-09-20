import numpy as np
import nibabel as nib


ORIGINAL_MASK = "input/prediction_mask.nii.gz"
POSTPROCESSED_MASK = "postprocessed/cleaned_prediction_mask.nii.gz"
GROUND_TRUTH_MASK = "input/ground_truth.nii.gz"


original = np.rint(
    nib.load(ORIGINAL_MASK).get_fdata()
).astype(np.uint8)

postprocessed = np.rint(
    nib.load(POSTPROCESSED_MASK).get_fdata()
).astype(np.uint8)

ground_truth = np.rint(
    nib.load(GROUND_TRUTH_MASK).get_fdata()
).astype(np.uint8)


label_names = {
    0: "Background",
    1: "NCR / NET",
    2: "Edema",
    3: "Enhancing Tumor"
}


print("=" * 60)
print("INDIVIDUAL LABEL VOXEL COUNT CHECK")
print("=" * 60)


for label, name in label_names.items():

    original_count = np.sum(
        original == label
    )

    postprocessed_count = np.sum(
        postprocessed == label
    )

    ground_truth_count = np.sum(
        ground_truth == label
    )

    print("\n" + name)
    print("-" * 40)

    print(
        "Original prediction:",
        original_count
    )

    print(
        "Post-processed prediction:",
        postprocessed_count
    )

    print(
        "Ground truth:",
        ground_truth_count
    )


print("\n" + "=" * 60)
print("CHECK COMPLETED")
print("=" * 60)