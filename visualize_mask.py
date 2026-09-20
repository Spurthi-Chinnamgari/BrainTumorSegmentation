import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


ORIGINAL_MASK = "input/prediction_mask.nii.gz"
POSTPROCESSED_MASK = "postprocessed/cleaned_prediction_mask.nii.gz"


print("Loading masks...")

original_nii = nib.load(ORIGINAL_MASK)
processed_nii = nib.load(POSTPROCESSED_MASK)

original = np.rint(original_nii.get_fdata()).astype(np.uint8)
processed = np.rint(processed_nii.get_fdata()).astype(np.uint8)

print("Original shape:", original.shape)
print("Post-processed shape:", processed.shape)

print("Original labels:", np.unique(original))
print("Post-processed labels:", np.unique(processed))


# Create a maximum-intensity projection along the Z direction.
# This uses the COMPLETE 3D volume rather than selecting one slice.

original_projection = np.max(original, axis=2)
processed_projection = np.max(processed, axis=2)


# Color map:
# 0 = black
# 1 = blue
# 2 = green
# 3 = red

colors = np.zeros((4, 3))

colors[0] = [0.0, 0.0, 0.0]   # Background
colors[1] = [0.0, 0.0, 1.0]   # NCR / NET
colors[2] = [0.0, 1.0, 0.0]   # Edema
colors[3] = [1.0, 0.0, 0.0]   # Enhancing Tumor


original_rgb = colors[original_projection]
processed_rgb = colors[processed_projection]


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(original_rgb)
axes[0].set_title("Original Prediction Mask")
axes[0].axis("off")

axes[1].imshow(processed_rgb)
axes[1].set_title("Post-Processed Mask")
axes[1].axis("off")


plt.tight_layout()
plt.show()