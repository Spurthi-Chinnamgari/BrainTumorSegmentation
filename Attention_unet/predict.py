"""
3D Attention U-Net Inference
============================

Uses the trained Attention U-Net model to generate a
3D segmentation mask for a REAL BraTS patient.

Training patch size:
    (16, 32, 32)

Input modalities:
    t1n, t1c, t2w, t2f

Output classes:
    0 = Background
    1 = NCR/NET
    2 = Edema
    3 = Enhancing Tumor
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

from dataset import BraTSDataset3D

from dataset.transforms import (
    ZScoreNormalize,
    ToTensor,
    ComposeTransforms,
)

from models import AttentionUNet3D

from utils import load_checkpoint


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

    print()
    print("=" * 60)
    print("LOADING PATIENT")
    print("=" * 60)

    print()
    print("Patient directory:")
    print(patient_dir)

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
    Input:
        (C, D, H, W)

    Output:
        padded image
    """

    if image.ndim != 4:
        raise RuntimeError(
            "pad_volume expected (C,D,H,W), "
            f"but got {tuple(image.shape)}"
        )

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

def get_positions(
    volume_size,
    patch_size,
    stride,
):

    if volume_size <= patch_size:
        return [0]

    positions = []

    position = 0

    while position + patch_size < volume_size:

        positions.append(position)

        position += stride

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

    if image.ndim != 5:

        raise RuntimeError(
            "Expected image shape "
            "(B,C,D,H,W), but got "
            f"{tuple(image.shape)}"
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
    # PATCH POSITIONS
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

    total_patches = (
        len(d_positions)
        * len(h_positions)
        * len(w_positions)
    )

    print()
    print("=" * 60)
    print("SLIDING-WINDOW INFERENCE")
    print("=" * 60)

    print()
    print("Volume :", (D, H, W))
    print("Patch  :", patch_size)
    print("Stride :", stride)

    print()
    print("D positions:", d_positions)
    print("H positions:", h_positions)
    print("W positions:", w_positions)

    print()
    print("Total patches:", total_patches)

    # --------------------------------------------------------
    # ACCUMULATORS
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
    # INFERENCE
    # --------------------------------------------------------

    model.eval()

    patch_number = 0

    with torch.no_grad():

        for d in d_positions:

            for h in h_positions:

                for w in w_positions:

                    patch_number += 1

                    patch = image[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw,
                    ]

                    patch = patch.to(device)

                    logits = model(patch)

                    if logits.ndim != 5:

                        raise RuntimeError(
                            "Model output should be "
                            "(B,C,D,H,W), but got "
                            f"{tuple(logits.shape)}"
                        )

                    if logits.shape[1] != NUM_CLASSES:

                        raise RuntimeError(
                            f"Expected {NUM_CLASSES} classes, "
                            f"but model returned "
                            f"{logits.shape[1]}"
                        )

                    # ------------------------------------------------
                    # ADD LOGITS
                    # ------------------------------------------------

                    logits_sum[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw,
                    ] += logits.float()

                    # ------------------------------------------------
                    # COUNT OVERLAPPING PATCHES
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
    # AVERAGE LOGITS
    # --------------------------------------------------------

    logits_average = (
        logits_sum /
        count_map.clamp_min(1.0)
    )

    # --------------------------------------------------------
    # ARGMAX
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

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # FIND T1N
    # --------------------------------------------------------

    t1_path = os.path.join(
        patient_dir,
        f"{patient_id}-t1n.nii.gz",
    )

    if not os.path.isfile(t1_path):

        t1_path = None

        for name in os.listdir(patient_dir):

            if (
                "t1n" in name.lower()
                and name.endswith(".nii.gz")
                and os.path.isfile(
                    os.path.join(patient_dir, name)
                )
            ):

                t1_path = os.path.join(
                    patient_dir,
                    name,
                )

                break

    if t1_path is None:

        raise FileNotFoundError(
            "Could not find T1 native MRI."
        )

    print()
    print("=" * 60)
    print("SAVING 3D SEGMENTATION")
    print("=" * 60)

    print()
    print("Reference MRI:")
    print(t1_path)

    reference_nii = nib.load(t1_path)

    if reference_nii.shape != prediction.shape:

        raise RuntimeError(
            "Prediction shape does not match MRI shape.\n"
            f"MRI: {reference_nii.shape}\n"
            f"Prediction: {prediction.shape}"
        )

    output_path = os.path.join(
        output_dir,
        f"{patient_id}_pred_seg.nii.gz",
    )

    prediction_nii = nib.Nifti1Image(
        prediction.astype(np.uint8),
        reference_nii.affine,
        reference_nii.header.copy(),
    )

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

    print()
    print("=" * 60)
    print("PREDICTION STATISTICS")
    print("=" * 60)

    print()
    print("Prediction shape:")
    print(prediction.shape)

    # --------------------------------------------------------
    # PREDICTION CLASSES
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
    # TUMOR VOXELS
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
    # GROUND TRUTH
    # --------------------------------------------------------

    if ground_truth is not None:

        if torch.is_tensor(ground_truth):

            gt = ground_truth.cpu().numpy()

        else:

            gt = np.asarray(
                ground_truth
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

        gt_tumor_voxels = int(
            np.sum(gt > 0)
        )

        print()
        print(
            "Ground Truth tumor voxels:",
            gt_tumor_voxels,
        )

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
        help="Trained model checkpoint",
    )

    parser.add_argument(
        "--patient-dir",
        type=str,
        default=(
            "C:/Datasets/BraTS2023/"
            "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
            "BraTS-GLI-00008-001"
        ),
        help="REAL BraTS patient directory",
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
    print("Device:")
    print(device)

    print()
    print("Patient:")
    print(args.patient_dir)

    print()
    print("Checkpoint:")
    print(args.checkpoint)

    # ========================================================
    # CHECK PATIENT
    # ========================================================

    if not os.path.isdir(
        args.patient_dir
    ):

        raise FileNotFoundError(
            "Patient directory does not exist:\n"
            f"{args.patient_dir}"
        )

    # ========================================================
    # CHECK CHECKPOINT
    # ========================================================

    if not os.path.isfile(
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
        dropout=0.1,
        use_transpose=True,
    ).to(device)

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print()
    print("Loading trained model...")

    load_checkpoint(
        args.checkpoint,
        model=model,
        device=device,
    )

    print()
    print("Checkpoint loaded successfully.")

    # ========================================================
    # LOAD PATIENT
    # ========================================================

    image, ground_truth = load_patient(
        args.patient_dir
    )

    print()
    print("Original image shape:")
    print(image.shape)

    if image.ndim != 4:

        raise RuntimeError(
            "Expected image shape "
            "(4,D,H,W), but got "
            f"{tuple(image.shape)}"
        )

    # ========================================================
    # PAD
    # ========================================================

    original_shape = image.shape[1:]

    print()
    print("Padding volume if necessary...")

    image_padded, padding = pad_volume(
        image,
        PATCH_SIZE,
    )

    print()
    print("Original spatial shape:")
    print(original_shape)

    print()
    print("Padded spatial shape:")
    print(image_padded.shape[1:])

    print()
    print("Padding:")
    print(padding)

    # ========================================================
    # ADD BATCH DIMENSION
    # ========================================================

    image_padded = image_padded.unsqueeze(0)

    print()
    print("Model input:")
    print(tuple(image_padded.shape))

    # ========================================================
    # PREDICTION
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
    print("Prediction shape after removing padding:")
    print(prediction.shape)

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    gt_np = None

    if ground_truth is not None:

        if torch.is_tensor(ground_truth):

            gt_np = ground_truth.cpu().numpy()

        else:

            gt_np = np.asarray(
                ground_truth
            )

        gt_np = np.squeeze(gt_np)

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
    print("Patient:")
    print(patient_id)

    print()
    print("Prediction saved to:")
    print(output_path)

    print()
    print("Final prediction shape:")
    print(prediction.shape)

    print()
    print("Prediction classes:")
    print(np.unique(prediction))

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