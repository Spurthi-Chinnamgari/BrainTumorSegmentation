"""
3D Attention U-Net Inference
============================

Inference using the processed BraTS 2023 dataset.

Processed input:
    images.npy -> (N, 4, 64, 64, 64)

Modalities:
    0 = T1n
    1 = T1c
    2 = T2w
    3 = T2f

Output classes:
    0 = Background
    1 = NCR/NET
    2 = Edema
    3 = Enhancing Tumor

IMPORTANT:
    This baseline inference operates on the already extracted
    64x64x64 processed patches.

    It does NOT:
        - load raw NIfTI for model input
        - perform Z-score normalization again
        - perform random cropping
        - use the old 16x32x32 patches
        - use the old 50% sliding-window inference

True full-volume reconstruction requires patch-coordinate /
cropped-volume metadata from preprocessing.
"""

import os
import argparse

import numpy as np
import torch

from dataset import BraTSDataset3D
from models import AttentionUNet3D
from utils import load_checkpoint


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = (64, 64, 64)

NUM_MODALITIES = 4
NUM_CLASSES = 4

FEATURES = [
    16,
    32,
    64,
    128,
    256,
]

DROPOUT = 0.1


# ============================================================
# PATH DISCOVERY
# ============================================================

def find_processed_root():

    candidates = [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "preprocessing",
                "data",
                "processed",
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "processed",
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "processed",
            )
        ),
    ]

    for path in candidates:

        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        "Could not find processed dataset.\n"
        "Expected one of:\n"
        + "\n".join(candidates)
    )


def find_split_root():

    candidates = [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "preprocessing",
                "data",
                "splits",
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "splits",
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "splits",
            )
        ),
    ]

    for path in candidates:

        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        "Could not find split directory.\n"
        "Expected one of:\n"
        + "\n".join(candidates)
    )


# ============================================================
# SPLIT FILE
# ============================================================

def read_patient_ids(split_file):

    if not os.path.isfile(split_file):

        raise FileNotFoundError(
            f"Split file does not exist:\n{split_file}"
        )

    patient_ids = []

    with open(
        split_file,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            patient_id = line.strip()

            if patient_id:
                patient_ids.append(patient_id)

    if not patient_ids:

        raise RuntimeError(
            f"Split file is empty:\n{split_file}"
        )

    return patient_ids


# ============================================================
# VERIFY PROCESSED PATIENT
# ============================================================

def verify_patient(
    processed_root,
    split,
    patient_id,
):

    patient_dir = os.path.join(
        processed_root,
        split,
        patient_id,
    )

    images_path = os.path.join(
        patient_dir,
        "images.npy",
    )

    masks_path = os.path.join(
        patient_dir,
        "masks.npy",
    )

    if not os.path.isfile(images_path):

        raise FileNotFoundError(
            f"Missing images.npy:\n{images_path}"
        )

    if not os.path.isfile(masks_path):

        raise FileNotFoundError(
            f"Missing masks.npy:\n{masks_path}"
        )

    images = np.load(
        images_path,
        mmap_mode="r",
    )

    masks = np.load(
        masks_path,
        mmap_mode="r",
    )

    if images.ndim != 5:

        raise RuntimeError(
            f"{patient_id}: expected images "
            f"(N,4,64,64,64), got {images.shape}"
        )

    if images.shape[1:] != (
        NUM_MODALITIES,
        *PATCH_SIZE,
    ):

        raise RuntimeError(
            f"{patient_id}: invalid image shape "
            f"{images.shape}. "
            f"Expected (N,4,64,64,64)."
        )

    if masks.ndim != 4:

        raise RuntimeError(
            f"{patient_id}: expected masks "
            f"(N,64,64,64), got {masks.shape}"
        )

    if masks.shape[1:] != PATCH_SIZE:

        raise RuntimeError(
            f"{patient_id}: invalid mask shape "
            f"{masks.shape}. "
            f"Expected (N,64,64,64)."
        )

    if images.shape[0] != masks.shape[0]:

        raise RuntimeError(
            f"{patient_id}: image/mask patch count mismatch.\n"
            f"Images: {images.shape[0]}\n"
            f"Masks : {masks.shape[0]}"
        )

    labels = np.unique(masks)

    invalid_labels = [
        int(x)
        for x in labels
        if int(x) not in range(NUM_CLASSES)
    ]

    if invalid_labels:

        raise RuntimeError(
            f"{patient_id}: invalid mask labels "
            f"{invalid_labels}. "
            f"Expected labels 0,1,2,3."
        )

    return images.shape[0]


# ============================================================
# PREDICT ONE PATCH
# ============================================================

def predict_patch(
    model,
    image,
    device,
):

    if image.ndim != 4:

        raise RuntimeError(
            "Expected image "
            "(C,D,H,W), got "
            f"{tuple(image.shape)}"
        )

    if tuple(image.shape) != (
        NUM_MODALITIES,
        *PATCH_SIZE,
    ):

        raise RuntimeError(
            "Invalid model input shape.\n"
            f"Got: {tuple(image.shape)}\n"
            f"Expected: "
            f"({NUM_MODALITIES},64,64,64)"
        )

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(image)

    expected_shape = (
        1,
        NUM_CLASSES,
        *PATCH_SIZE,
    )

    if tuple(logits.shape) != expected_shape:

        raise RuntimeError(
            "Invalid model output shape.\n"
            f"Got: {tuple(logits.shape)}\n"
            f"Expected: {expected_shape}"
        )

    prediction = torch.argmax(
        logits,
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

    tumor_voxels = int(
        np.sum(prediction > 0)
    )

    print()
    print(
        "Predicted tumor voxels:",
        tumor_voxels,
    )

    if ground_truth is not None:

        gt = np.asarray(
            ground_truth
        ).squeeze()

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
# PREDICT PROCESSED PATIENT
# ============================================================

def predict_patient(
    model,
    dataset,
    patient_id,
    device,
):

    print()
    print("=" * 60)
    print("PATIENT INFERENCE")
    print("=" * 60)

    print()
    print("Patient:")
    print(patient_id)

    print()
    print("Number of processed patches:")
    print(len(dataset))

    predictions = []

    ground_truths = []

    model.eval()

    for index in range(len(dataset)):

        sample = dataset[index]

        image = sample["image"]
        mask = sample.get("mask")

        prediction = predict_patch(
            model=model,
            image=image,
            device=device,
        )

        predictions.append(
            prediction
        )

        if mask is not None:

            if torch.is_tensor(mask):

                mask_np = (
                    mask.cpu()
                    .numpy()
                    .astype(np.uint8)
                )

            else:

                mask_np = np.asarray(
                    mask,
                    dtype=np.uint8,
                )

            ground_truths.append(
                mask_np
            )

        print(
            f"\rProcessing patch "
            f"{index + 1}/{len(dataset)}",
            end="",
        )

    print()

    predictions = np.stack(
        predictions,
        axis=0,
    )

    print()
    print("Prediction array shape:")
    print(predictions.shape)

    return predictions, ground_truths


# ============================================================
# SAVE PROCESSED PREDICTION
# ============================================================

def save_prediction(
    predictions,
    ground_truths,
    output_dir,
    patient_id,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    prediction_path = os.path.join(
        output_dir,
        f"{patient_id}_predictions.npy",
    )

    np.save(
        prediction_path,
        predictions,
    )

    print()
    print("Prediction saved:")
    print(prediction_path)

    if ground_truths:

        ground_truth = np.stack(
            ground_truths,
            axis=0,
        )

        ground_truth_path = os.path.join(
            output_dir,
            f"{patient_id}_ground_truth.npy",
        )

        np.save(
            ground_truth_path,
            ground_truth,
        )

        print()
        print("Ground truth saved:")
        print(ground_truth_path)

    return prediction_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "3D Attention U-Net inference "
            "on processed BraTS patches"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pth",
        help="Trained model checkpoint",
    )

    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help=(
            "Patient ID from the test split. "
            "If omitted, the first test patient is used."
        ),
    )

    parser.add_argument(
        "--processed-root",
        type=str,
        default=None,
        help="Processed dataset root",
    )

    parser.add_argument(
        "--split-root",
        type=str,
        default=None,
        help="Split directory",
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

    # ========================================================
    # PATHS
    # ========================================================

    processed_root = (
        os.path.abspath(args.processed_root)
        if args.processed_root
        else find_processed_root()
    )

    split_root = (
        os.path.abspath(args.split_root)
        if args.split_root
        else find_split_root()
    )

    split_file = os.path.join(
        split_root,
        "test.txt",
    )

    print()
    print("Processed dataset:")
    print(processed_root)

    print()
    print("Test split:")
    print(split_file)

    # ========================================================
    # CHECK CHECKPOINT
    # ========================================================

    checkpoint = os.path.abspath(
        args.checkpoint
    )

    if not os.path.isfile(checkpoint):

        raise FileNotFoundError(
            "Checkpoint does not exist:\n"
            f"{checkpoint}\n\n"
            "Train the model first before running inference."
        )

    # ========================================================
    # READ TEST SPLIT
    # ========================================================

    patient_ids = read_patient_ids(
        split_file
    )

    print()
    print("Test patients:")
    print(patient_ids)

    # ========================================================
    # SELECT PATIENT
    # ========================================================

    if args.patient_id is None:

        patient_id = patient_ids[0]

    else:

        patient_id = args.patient_id

        if patient_id not in patient_ids:

            raise ValueError(
                f"Patient '{patient_id}' is not present "
                "in test.txt."
            )

    # ========================================================
    # VERIFY PATIENT
    # ========================================================

    num_patches = verify_patient(
        processed_root=processed_root,
        split="test",
        patient_id=patient_id,
    )

    print()
    print("Patient:")
    print(patient_id)

    print()
    print("Processed patches:")
    print(num_patches)

    # ========================================================
    # CREATE DATASET
    # ========================================================

    dataset = BraTSDataset3D(
        processed_root=processed_root,
        split="test",
        split_file=split_file,
        patch_size=PATCH_SIZE,
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print()
    print("Creating Attention U-Net 3D...")

    model = AttentionUNet3D(
        in_channels=NUM_MODALITIES,
        out_channels=NUM_CLASSES,
        features=FEATURES,
        dropout=DROPOUT,
        use_transpose=True,
    ).to(device)

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print()
    print("Loading trained checkpoint...")

    load_checkpoint(
        checkpoint,
        model=model,
        device=device,
    )

    print()
    print("Checkpoint loaded successfully.")

    # ========================================================
    # GET PATIENT PATCH INDICES
    # ========================================================

    patient_indices = [
        index
        for index in range(len(dataset))
        if dataset[index]["patient_id"] == patient_id
    ]

    if not patient_indices:

        raise RuntimeError(
            f"No processed patches found for "
            f"patient {patient_id}."
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = []
    ground_truths = []

    model.eval()

    for count, dataset_index in enumerate(
        patient_indices,
        start=1,
    ):

        sample = dataset[
            dataset_index
        ]

        image = sample["image"]
        mask = sample["mask"]

        prediction = predict_patch(
            model=model,
            image=image,
            device=device,
        )

        predictions.append(
            prediction
        )

        if torch.is_tensor(mask):

            mask_np = (
                mask.cpu()
                .numpy()
                .astype(np.uint8)
            )

        else:

            mask_np = np.asarray(
                mask,
                dtype=np.uint8,
            )

        ground_truths.append(
            mask_np
        )

        print(
            f"\rProcessing patch "
            f"{count}/{len(patient_indices)}",
            end="",
        )

    print()

    predictions = np.stack(
        predictions,
        axis=0,
    )

    ground_truths = np.stack(
        ground_truths,
        axis=0,
    )

    # ========================================================
    # PATCH STATISTICS
    # ========================================================

    print()
    print("=" * 60)
    print("PATIENT PATCH RESULTS")
    print("=" * 60)

    print()
    print("Prediction array:")
    print(predictions.shape)

    print()
    print("Ground truth array:")
    print(ground_truths.shape)

    print()

    for index in range(
        len(predictions)
    ):

        print(
            f"Patch {index + 1}:"
        )

        print_statistics(
            predictions[index],
            ground_truths[index],
        )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = save_prediction(
        predictions=predictions,
        ground_truths=ground_truths,
        output_dir=args.output_dir,
        patient_id=patient_id,
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
    print("Prediction:")
    print(output_path)

    print()
    print("Prediction shape:")
    print(predictions.shape)

    print()
    print("Classes:")
    print(np.unique(predictions))

    print()
    print(
        "Predicted tumor voxels:",
        int(np.sum(predictions > 0)),
    )

    print()
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()