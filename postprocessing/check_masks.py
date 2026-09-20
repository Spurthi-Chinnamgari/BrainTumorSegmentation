import numpy as np
import nibabel as nib


PREDICTION_MASK = "input/prediction_mask.nii.gz"
GROUND_TRUTH_MASK = "input/ground_truth.nii.gz"


print("=" * 60)
print("CHECKING PREDICTION AND GROUND-TRUTH MASKS")
print("=" * 60)


# Load prediction mask
print("\nLoading prediction mask...")
prediction_nii = nib.load(PREDICTION_MASK)
prediction = np.rint(
    prediction_nii.get_fdata()
).astype(np.uint8)


# Load ground-truth mask
print("Loading ground-truth mask...")
ground_truth_nii = nib.load(GROUND_TRUTH_MASK)
ground_truth = np.rint(
    ground_truth_nii.get_fdata()
).astype(np.uint8)


# Display shapes
print("\nPrediction shape:")
print(prediction.shape)

print("\nGround-truth shape:")
print(ground_truth.shape)


# Display labels
print("\nPrediction labels:")
print(np.unique(prediction))

print("\nGround-truth labels:")
print(np.unique(ground_truth))


# Check dimensions
print("\nChecking dimensions...")

if prediction.shape == ground_truth.shape:
    print("PASS: Both masks have the same 3D shape.")
else:
    print("ERROR: The masks have different 3D shapes.")


# Check voxel spacing
prediction_spacing = prediction_nii.header.get_zooms()[:3]
ground_truth_spacing = ground_truth_nii.header.get_zooms()[:3]

print("\nPrediction voxel spacing:")
print(prediction_spacing)

print("\nGround-truth voxel spacing:")
print(ground_truth_spacing)


# Check affine matrices
print("\nChecking affine matrices...")

if np.allclose(
    prediction_nii.affine,
    ground_truth_nii.affine
):
    print("PASS: Affine matrices are compatible.")
else:
    print("WARNING: Affine matrices are different.")


print("\n" + "=" * 60)
print("MASK CHECK COMPLETED")
print("=" * 60)