"""
3D Attention U-Net Inference Pipeline
-------------------------------------

Uses the SAME 3D patch size used during training:

    Training patch = (16, 32, 32)

Inference is performed using overlapping patches and
the logits are averaged before taking argmax.

IMPORTANT:
    pad_volume() works on (C, D, H, W).
    The batch dimension is added AFTER padding.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

from dataset import (
    BraTSDataset3D,
    ZScoreNormalize,
    ToTensor,
    ComposeTransforms,
)

from models import AttentionUNet3D

from utils import (
    load_checkpoint,
)


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = (16, 32, 32)

# 50% overlap
STRIDE = (8, 16, 16)

NUM_CLASSES = 4


# ============================================================
# LOAD PATIENT
# ============================================================

def load_patient(patient_dir):
    """
    Load patient using the SAME dataset preprocessing
    used during training.

    Returns:

        image:
            (4, D, H, W)

        mask:
            (D, H, W)
    """

    transform = ComposeTransforms([
        ZScoreNormalize(),
        ToTensor(),
    ])

    dataset = BraTSDataset3D(
        [patient_dir],
        transform=transform,
        is_train=False,
    )

    sample = dataset[0]

    image = sample["image"]

    mask = sample.get("mask", None)

    return image, mask


# ============================================================
# PAD VOLUME
# ============================================================

def pad_volume(image, patch_size):
    """
    Make sure the volume is at least as large as the patch.

    IMPORTANT:
        Input MUST be:

            (C, D, H, W)

        NOT:

            (B, C, D, H, W)
    """

    _, D, H, W = image.shape

    pd = max(0, patch_size[0] - D)
    ph = max(0, patch_size[1] - H)
    pw = max(0, patch_size[2] - W)

    if pd == 0 and ph == 0 and pw == 0:
        return image, (0, 0, 0)

    image = F.pad(
        image,
        (
            0, pw,
            0, ph,
            0, pd,
        ),
        mode="constant",
        value=0,
    )

    return image, (pd, ph, pw)


# ============================================================
# PATCH POSITIONS
# ============================================================

def get_positions(volume_size, patch_size, stride):
    """
    Generate patch starting positions.

    The last patch is forced to reach the end
    of the volume.
    """

    if volume_size <= patch_size:
        return [0]

    positions = []

    pos = 0

    while pos + patch_size < volume_size:

        positions.append(pos)

        pos += stride

    last_position = volume_size - patch_size

    if len(positions) == 0 or positions[-1] != last_position:

        positions.append(last_position)

    return positions


# ============================================================
# SLIDING WINDOW PREDICTION
# ============================================================

def sliding_window_prediction(
    model,
    image,
    device,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
):
    """
    Perform overlapping 3D patch inference.

    Input:

        image:
            (1, 4, D, H, W)

    Output:

        prediction:
            (D, H, W)
    """

    # --------------------------------------------------------
    # Check input shape
    # --------------------------------------------------------

    if image.ndim != 5:

        raise RuntimeError(
            "sliding_window_prediction expected "
            f"(B,C,D,H,W), but got {tuple(image.shape)}"
        )

    B, C, D, H, W = image.shape

    if B != 1:

        raise RuntimeError(
            f"Expected batch size 1, got {B}"
        )

    if C != 4:

        raise RuntimeError(
            f"Expected 4 MRI modalities, got {C}"
        )

    pd, ph, pw = patch_size

    # --------------------------------------------------------
    # Get positions
    # --------------------------------------------------------

    d_positions = get_positions(
        D,
        pd,
        stride[0],
    )

    h_positions = get_positions(
        H,
        ph,
        stride[1],
    )

    w_positions = get_positions(
        W,
        pw,
        stride[2],
    )

    print()
    print("=" * 60)
    print("SLIDING-WINDOW INFERENCE")
    print("=" * 60)

    print("Volume :", (D, H, W))
    print("Patch  :", patch_size)
    print("Stride :", stride)

    print("D positions:", d_positions)
    print("H positions:", h_positions)
    print("W positions:", w_positions)

    total_patches = (
        len(d_positions)
        * len(h_positions)
        * len(w_positions)
    )

    print("Total patches:", total_patches)

    # --------------------------------------------------------
    # Allocate accumulators
    # --------------------------------------------------------

    logits_sum = torch.zeros(
        (
            1,
            NUM_CLASSES,
            D,
            H,
            W,
        ),
        dtype=torch.float32,
        device=device,
    )

    count_map = torch.zeros(
        (
            1,
            1,
            D,
            H,
            W,
        ),
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    model.eval()

    patch_number = 0

    with torch.no_grad():

        for d in d_positions:

            for h in h_positions:

                for w in w_positions:

                    patch_number += 1

                    # ------------------------------------------------
                    # Extract patch
                    # ------------------------------------------------

                    patch = image[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw,
                    ]

                    patch = patch.to(device)

                    # ------------------------------------------------
                    # Model
                    # ------------------------------------------------

                    logits = model(patch)

                    # ------------------------------------------------
                    # Validate output
                    # ------------------------------------------------

                    if logits.ndim != 5:

                        raise RuntimeError(
                            "Model output should be "
                            "(B,C,D,H,W), but got "
                            f"{tuple(logits.shape)}"
                        )

                    if logits.shape[1] != NUM_CLASSES:

                        raise RuntimeError(
                            f"Expected {NUM_CLASSES} output classes, "
                            f"but model returned "
                            f"{logits.shape[1]}"
                        )

                    # ------------------------------------------------
                    # Accumulate logits
                    # ------------------------------------------------

                    logits_sum[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw,
                    ] += logits.float()

                    # ------------------------------------------------
                    # Count how many predictions cover each voxel
                    # ------------------------------------------------

                    count_map[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw,
                    ] += 1.0

                    print(
                        f"\rProcessing patch "
                        f"{patch_number}/{total_patches}",
                        end="",
                    )

    print()

    # --------------------------------------------------------
    # Average overlapping logits
    # --------------------------------------------------------

    logits_average = (
        logits_sum /
        count_map.clamp_min(1.0)
    )

    # --------------------------------------------------------
    # Argmax
    # --------------------------------------------------------

    prediction = torch.argmax(
        logits_average,
        dim=1,
    )

    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return prediction


# ============================================================
# SAVE NIFTI
# ============================================================

def save_prediction_nifti(
    prediction,
    patient_dir,
    patient_id,
    output_dir,
):
    """
    Save prediction as NIfTI.

    Uses the original T1 MRI affine/header
    so the prediction remains spatially aligned.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find T1 native MRI
    # --------------------------------------------------------

    t1_path = os.path.join(
        patient_dir,
        f"{patient_id}-t1n.nii.gz",
    )

    # --------------------------------------------------------
    # Fallback search
    # --------------------------------------------------------

    if not os.path.exists(t1_path):

        for name in os.listdir(patient_dir):

            if (
                "t1n" in name.lower()
                and name.endswith(".nii.gz")
            ):

                t1_path = os.path.join(
                    patient_dir,
                    name,
                )

                break

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if not os.path.exists(t1_path):

        raise FileNotFoundError(
            "Could not find T1 native MRI for affine."
        )

    print()
    print("=" * 60)
    print("SAVING PREDICTION")
    print("=" * 60)

    print()
    print("Reference MRI:")
    print(t1_path)

    # --------------------------------------------------------
    # Load reference MRI
    # --------------------------------------------------------

    reference_nii = nib.load(
        t1_path
    )

    # --------------------------------------------------------
    # Check shape
    # --------------------------------------------------------

    if reference_nii.shape != prediction.shape:

        raise RuntimeError(
            "Prediction shape does not match "
            "reference MRI shape.\n"
            f"Reference: {reference_nii.shape}\n"
            f"Prediction: {prediction.shape}"
        )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    output_path = os.path.join(
        output_dir,
        f"{patient_id}_pred_seg.nii.gz",
    )

    # --------------------------------------------------------
    # Create NIfTI
    # --------------------------------------------------------

    prediction_nii = nib.Nifti1Image(
        prediction.astype(np.uint8),
        reference_nii.affine,
        reference_nii.header.copy(),
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    nib.save(
        prediction_nii,
        output_path,
    )

    return output_path


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    prediction,
    ground_truth=None,
):
    """
    Print prediction and ground-truth statistics.
    """

    print()
    print("=" * 60)
    print("PREDICTION STATISTICS")
    print("=" * 60)

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    print()
    print(
        "Prediction shape:",
        prediction.shape,
    )

    # --------------------------------------------------------
    # Prediction classes
    # --------------------------------------------------------

    unique, counts = np.unique(
        prediction,
        return_counts=True,
    )

    print()
    print("Prediction classes:")

    for value, count in zip(
        unique,
        counts,
    ):

        print(
            f"  Class {int(value)}: "
            f"{int(count)} voxels"
        )

    # --------------------------------------------------------
    # Tumor voxels
    # --------------------------------------------------------

    tumor_voxels = int(
        np.sum(prediction > 0)
    )

    print()
    print(
        "Predicted tumor voxels:",
        tumor_voxels,
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    if ground_truth is not None:

        gt = np.asarray(
            ground_truth,
            dtype=np.int64,
        )

        gt = np.squeeze(gt)

        print()
        print("Ground Truth shape:")

        print(gt.shape)

        gt_unique, gt_counts = np.unique(
            gt,
            return_counts=True,
        )

        print()
        print("Ground Truth classes:")

        for value, count in zip(
            gt_unique,
            gt_counts,
        ):

            print(
                f"  Class {int(value)}: "
                f"{int(count)} voxels"
            )

        # ----------------------------------------------------
        # Ground truth tumor
        # ----------------------------------------------------

        gt_tumor_voxels = int(
            np.sum(gt > 0)
        )

        print()
        print(
            "Ground Truth tumor voxels:",
            gt_tumor_voxels,
        )

        # ----------------------------------------------------
        # Agreement
        # ----------------------------------------------------

        if gt.shape == prediction.shape:

            agreement = np.mean(
                prediction == gt
            )

            print()
            print(
                f"Voxel agreement: "
                f"{agreement:.6f}"
            )

    print()
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser(
        description="3D Attention U-Net inference"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pth",
        help="Path to trained checkpoint",
    )

    parser.add_argument(
        "--patient-dir",
        type=str,
        default=(
            "./data/BraTS2023_Synthetic/"
            "BraTS2023_Synthetic_000"
        ),
        help="Patient directory",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/predictions",
        help="Output directory",
    )

    args = parser.parse_args()

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print("ATTENTION U-NET 3D INFERENCE")
    print("=" * 60)

    print()
    print("Device:", device)
    print("Patient:", args.patient_dir)
    print("Checkpoint:", args.checkpoint)

    # ========================================================
    # CHECK PATIENT
    # ========================================================

    if not os.path.exists(
        args.patient_dir
    ):

        raise FileNotFoundError(
            "Patient directory does not exist:\n"
            f"{args.patient_dir}"
        )

    # ========================================================
    # CHECK CHECKPOINT
    # ========================================================

    if not os.path.exists(
        args.checkpoint
    ):

        raise FileNotFoundError(
            "Checkpoint does not exist:\n"
            f"{args.checkpoint}"
        )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print()
    print("Creating Attention U-Net 3D...")

    model = AttentionUNet3D(
        in_channels=4,
        out_channels=4,
        features=[
            16,
            32,
            64,
            128,
            256,
        ],
    ).to(device)

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print()
    print("Loading checkpoint...")

    load_checkpoint(
        args.checkpoint,
        model=model,
        device=device,
    )

    print(
        "Checkpoint loaded successfully."
    )

    # ========================================================
    # LOAD PATIENT
    # ========================================================

    print()
    print("Loading patient MRI...")

    image, ground_truth = load_patient(
        args.patient_dir
    )

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    print()
    print(
        "Original image tensor shape:"
    )

    print(image.shape)

    if image.ndim != 4:

        raise RuntimeError(
            "Expected image shape "
            "(4,D,H,W), but got "
            f"{tuple(image.shape)}"
        )

    # ========================================================
    # IMPORTANT:
    #
    # PAD BEFORE ADDING BATCH DIMENSION
    #
    # image = (4,D,H,W)
    #
    # pad_volume() expects exactly this.
    # ========================================================

    print()
    print("Padding volume if necessary...")

    original_shape = image.shape[1:]

    image_padded, padding = pad_volume(
        image,
        PATCH_SIZE,
    )

    print(
        "Original spatial shape:",
        original_shape,
    )

    print(
        "Padded spatial shape:",
        image_padded.shape[1:],
    )

    print(
        "Padding:",
        padding,
    )

    # ========================================================
    # NOW ADD BATCH DIMENSION
    #
    # (4,D,H,W)
    #
    #       ↓
    #
    # (1,4,D,H,W)
    # ========================================================

    image_padded = image_padded.unsqueeze(
        0
    )

    print()
    print(
        "Model input volume:",
        tuple(image_padded.shape),
    )

    # ========================================================
    # SLIDING WINDOW
    # ========================================================

    prediction = sliding_window_prediction(
        model=model,
        image=image_padded,
        device=device,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )

    # ========================================================
    # REMOVE PADDING
    # ========================================================

    D, H, W = original_shape

    prediction = prediction[
        :D,
        :H,
        :W,
    ]

    print()
    print(
        "Prediction after removing padding:",
        prediction.shape,
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    gt_np = None

    if ground_truth is not None:

        if torch.is_tensor(
            ground_truth
        ):

            gt_np = ground_truth.cpu().numpy()

        else:

            gt_np = np.asarray(
                ground_truth
            )

        gt_np = np.squeeze(
            gt_np
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    print_statistics(
        prediction,
        gt_np,
    )

    # ========================================================
    # PATIENT ID
    # ========================================================

    patient_id = os.path.basename(
        os.path.normpath(
            args.patient_dir
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = save_prediction_nifti(
        prediction=prediction,
        patient_dir=args.patient_dir,
        patient_id=patient_id,
        output_dir=args.output_dir,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 60)
    print("INFERENCE SUCCESSFUL")
    print("=" * 60)

    print()
    print("Prediction saved to:")

    print(output_path)

    print()
    print("Final prediction shape:")

    print(prediction.shape)

    print()
    print("Final prediction classes:")

    print(
        np.unique(prediction)
    )

    print()
    print(
        "Predicted tumor voxels:",
        int(np.sum(prediction > 0)),
    )

    print()
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()