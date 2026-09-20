import numpy as np
import nibabel as nib
from scipy import ndimage


PREDICTION_MASK = "postprocessed/cleaned_prediction_mask.nii.gz"
GROUND_TRUTH_MASK = "input/ground_truth.nii.gz"


print("=" * 60)
print("BRAIN TUMOR SEGMENTATION PERFORMANCE METRICS")
print("=" * 60)


print("\nLoading masks...")

prediction_nii = nib.load(PREDICTION_MASK)
ground_truth_nii = nib.load(GROUND_TRUTH_MASK)

prediction = np.rint(
    prediction_nii.get_fdata()
).astype(np.uint8)

ground_truth = np.rint(
    ground_truth_nii.get_fdata()
).astype(np.uint8)


print("Prediction shape:", prediction.shape)
print("Ground-truth shape:", ground_truth.shape)


def dice_score(pred, gt):
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    if pred_sum == 0 and gt_sum == 0:
        return 1.0

    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = np.sum(pred & gt)

    return (
        2.0 * intersection /
        (pred_sum + gt_sum)
    )


def iou_score(pred, gt):
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)

    if union == 0:
        return 1.0

    return intersection / union


def precision_score(pred, gt):
    true_positive = np.sum(pred & gt)
    false_positive = np.sum(pred & ~gt)

    denominator = true_positive + false_positive

    if denominator == 0:
        return 1.0 if np.sum(gt) == 0 else 0.0

    return true_positive / denominator


def recall_score(pred, gt):
    true_positive = np.sum(pred & gt)
    false_negative = np.sum(~pred & gt)

    denominator = true_positive + false_negative

    if denominator == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0

    return true_positive / denominator


def surface_mask(mask):
    structure = ndimage.generate_binary_structure(
        3,
        1
    )

    eroded = ndimage.binary_erosion(
        mask,
        structure=structure,
        border_value=0
    )

    return mask ^ eroded


def hd95(pred, gt, spacing):
    if not np.any(pred) and not np.any(gt):
        return 0.0

    if not np.any(pred) or not np.any(gt):
        return np.nan

    pred_surface = surface_mask(pred)
    gt_surface = surface_mask(gt)

    pred_distance = ndimage.distance_transform_edt(
        ~pred_surface,
        sampling=spacing
    )

    gt_distance = ndimage.distance_transform_edt(
        ~gt_surface,
        sampling=spacing
    )

    pred_to_gt = pred_distance[gt_surface]
    gt_to_pred = gt_distance[pred_surface]

    all_distances = np.concatenate(
        [pred_to_gt, gt_to_pred]
    )

    return np.percentile(
        all_distances,
        95
    )


def asd(pred, gt, spacing):
    if not np.any(pred) and not np.any(gt):
        return 0.0

    if not np.any(pred) or not np.any(gt):
        return np.nan

    pred_surface = surface_mask(pred)
    gt_surface = surface_mask(gt)

    pred_distance = ndimage.distance_transform_edt(
        ~pred_surface,
        sampling=spacing
    )

    gt_distance = ndimage.distance_transform_edt(
        ~gt_surface,
        sampling=spacing
    )

    pred_to_gt = pred_distance[gt_surface]
    gt_to_pred = gt_distance[pred_surface]

    all_distances = np.concatenate(
        [pred_to_gt, gt_to_pred]
    )

    return np.mean(all_distances)


spacing = prediction_nii.header.get_zooms()[:3]


regions = {
    "Whole Tumor (WT)": [1, 2, 3],
    "Tumor Core (TC)": [1, 3],
    "Enhancing Tumor (ET)": [3]
}


print("\n" + "=" * 60)
print("METRIC RESULTS")
print("=" * 60)


for region_name, labels in regions.items():

    prediction_region = np.isin(
        prediction,
        labels
    )

    ground_truth_region = np.isin(
        ground_truth,
        labels
    )

    dice = dice_score(
        prediction_region,
        ground_truth_region
    )

    iou = iou_score(
        prediction_region,
        ground_truth_region
    )

    precision = precision_score(
        prediction_region,
        ground_truth_region
    )

    recall = recall_score(
        prediction_region,
        ground_truth_region
    )

    hd = hd95(
        prediction_region,
        ground_truth_region,
        spacing
    )

    average_surface_distance = asd(
        prediction_region,
        ground_truth_region,
        spacing
    )

    print("\n" + region_name)
    print("-" * 40)
    print(f"Dice Score : {dice:.4f}")
    print(f"IoU        : {iou:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")

    if np.isnan(hd):
        print("HD95       : N/A")
    else:
        print(f"HD95       : {hd:.4f} mm")

    if np.isnan(average_surface_distance):
        print("ASD        : N/A")
    else:
        print(
            f"ASD        : "
            f"{average_surface_distance:.4f} mm"
        )


print("\n" + "=" * 60)
print("METRIC CALCULATION COMPLETED")
print("=" * 60)