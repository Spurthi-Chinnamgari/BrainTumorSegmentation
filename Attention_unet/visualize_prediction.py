"""
BraTS 2023 - Attention U-Net 3D Prediction + 3D NIfTI Output
--------------------------------------------------------------

This script:

1. Loads T1n, T1c, T2w and T2f
2. Loads the trained 3D Attention U-Net
3. Performs FULL 3D volume prediction
4. Saves the predicted segmentation as:

       results/segmentation_3d.nii.gz

5. Also creates the familiar visualization:

       T1n | T1c | T2w | T2f | Segmentation

The NIfTI file is the important output for further 3D processing.
"""

import os
import glob
import argparse

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from models import AttentionUNet3D


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = (16, 32, 32)

# Stride controls overlap between patches.
# Smaller stride = more overlap = slower but smoother prediction.
STRIDE = (8, 16, 16)

NUM_CLASSES = 4

MODALITIES = {
    "T1n": "t1n",
    "T1c": "t1c",
    "T2w": "t2w",
    "T2f": "t2f",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FIND BRATS FILE
# ============================================================

def get_modality_file(patient_dir, modality):

    patient_dir = os.path.abspath(patient_dir)

    if not os.path.isdir(patient_dir):
        raise FileNotFoundError(
            f"\nPatient directory does not exist:\n"
            f"{patient_dir}\n"
        )

    patterns = [
        os.path.join(
            patient_dir,
            f"*-{modality}.nii.gz"
        ),

        os.path.join(
            patient_dir,
            f"*_{modality}.nii.gz"
        ),

        os.path.join(
            patient_dir,
            f"*{modality}.nii.gz"
        ),
    ]

    for pattern in patterns:

        matches = glob.glob(pattern)

        if len(matches) > 0:
            return matches[0]

    # Recursive search
    recursive_pattern = os.path.join(
        patient_dir,
        "**",
        f"*{modality}.nii.gz"
    )

    matches = glob.glob(
        recursive_pattern,
        recursive=True
    )

    if len(matches) > 0:
        return matches[0]

    raise FileNotFoundError(
        f"\nCould not find *{modality}.nii.gz "
        f"inside:\n{patient_dir}\n"
    )


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(path):

    nii = nib.load(path)

    data = nii.get_fdata()

    data = np.asarray(
        data,
        dtype=np.float32
    )

    return data


# ============================================================
# Z-SCORE NORMALIZATION
# ============================================================

def zscore_normalize(volume):

    volume = volume.astype(
        np.float32,
        copy=False
    )

    mask = volume != 0

    if not np.any(mask):
        return volume

    values = volume[mask]

    mean = values.mean()
    std = values.std()

    if std < 1e-8:

        volume[mask] = 0.0

    else:

        volume[mask] = (
            volume[mask] - mean
        ) / std

    return volume


# ============================================================
# LOAD PATIENT
# ============================================================

def load_patient(patient_dir):

    print()
    print("=" * 70)
    print("LOADING PATIENT")
    print("=" * 70)

    print(
        "Patient directory:",
        patient_dir
    )

    images = {}

    for display_name, modality in MODALITIES.items():

        path = get_modality_file(
            patient_dir,
            modality
        )

        print(
            f"{display_name}: {path}"
        )

        image = load_nifti(path)

        image = zscore_normalize(image)

        images[display_name] = image

    shapes = [
        image.shape
        for image in images.values()
    ]

    if len(set(shapes)) != 1:

        raise RuntimeError(
            "MRI modalities do not have "
            "the same shape.\n"
            f"Shapes found: {shapes}"
        )

    print()
    print(
        "Volume shape:",
        shapes[0]
    )

    return images


# ============================================================
# GET REFERENCE NIFTI
# ============================================================

def get_reference_nifti(patient_dir):

    """
    Use T1n as the reference image.

    Its affine/header are used when saving
    the predicted 3D segmentation.
    """

    t1n_path = get_modality_file(
        patient_dir,
        "t1n"
    )

    reference = nib.load(
        t1n_path
    )

    return reference


# ============================================================
# LOAD GROUND TRUTH IF AVAILABLE
# ============================================================

def load_ground_truth(patient_dir):

    try:

        seg_path = get_modality_file(
            patient_dir,
            "seg"
        )

        print(
            "Ground-truth segmentation:",
            seg_path
        )

        seg = load_nifti(seg_path)

        return seg.astype(
            np.int64
        )

    except FileNotFoundError:

        print(
            "Ground-truth segmentation not found."
        )

        return None


# ============================================================
# MODEL
# ============================================================

def create_model():

    print()
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

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
    )

    model = model.to(
        DEVICE
    )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path
):

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{checkpoint_path}\n"
        )

    print(
        "Checkpoint:",
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model" in checkpoint:

            state_dict = checkpoint[
                "model"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove DataParallel "module." prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )
    )

    if len(missing) > 0:

        print(
            "\nWARNING: Missing keys:",
            len(missing)
        )

        for key in missing[:10]:

            print(
                "   ",
                key
            )

    if len(unexpected) > 0:

        print(
            "\nWARNING: Unexpected keys:",
            len(unexpected)
        )

        for key in unexpected[:10]:

            print(
                "   ",
                key
            )

    model.eval()

    print()
    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(images):

    volume_list = [
        images["T1n"],
        images["T1c"],
        images["T2w"],
        images["T2f"],
    ]

    volume = np.stack(
        volume_list,
        axis=0
    )

    tensor = torch.from_numpy(
        volume
    ).float()

    tensor = tensor.unsqueeze(
        0
    )

    return tensor


# ============================================================
# CALCULATE PATCH START POSITIONS
# ============================================================

def get_positions(
    size,
    patch_size,
    stride
):

    if size <= patch_size:

        return [0]

    positions = []

    position = 0

    while position + patch_size < size:

        positions.append(
            position
        )

        position += stride

    last_position = (
        size - patch_size
    )

    if len(positions) == 0:

        positions.append(
            0
        )

    elif positions[-1] != last_position:

        positions.append(
            last_position
        )

    return positions


# ============================================================
# FULL 3D PREDICTION
# ============================================================

def predict_volume(
    model,
    volume
):

    """
    Perform FULL 3D sliding-window prediction.

    Input:

        (1, 4, D, H, W)

    Output:

        (D, H, W)

    Every voxel in the complete 3D brain volume
    receives a segmentation class.
    """

    print()
    print("=" * 70)
    print("FULL 3D PREDICTION")
    print("=" * 70)

    _, channels, depth, height, width = (
        volume.shape
    )

    pd, ph, pw = PATCH_SIZE

    sd, sh, sw = STRIDE

    print(
        "Input volume:",
        (depth, height, width)
    )

    print(
        "Patch size:",
        PATCH_SIZE
    )

    print(
        "Patch stride:",
        STRIDE
    )

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    pad_d = max(
        0,
        pd - depth
    )

    pad_h = max(
        0,
        ph - height
    )

    pad_w = max(
        0,
        pw - width
    )

    if (
        pad_d > 0
        or pad_h > 0
        or pad_w > 0
    ):

        volume = F.pad(
            volume,
            (
                0,
                pad_w,
                0,
                pad_h,
                0,
                pad_d,
            ),
            mode="constant",
            value=0
        )

    _, _, D, H, W = (
        volume.shape
    )

    # --------------------------------------------------------
    # Patch positions
    # --------------------------------------------------------

    d_positions = get_positions(
        D,
        pd,
        sd
    )

    h_positions = get_positions(
        H,
        ph,
        sh
    )

    w_positions = get_positions(
        W,
        pw,
        sw
    )

    total_patches = (
        len(d_positions)
        *
        len(h_positions)
        *
        len(w_positions)
    )

    print()
    print(
        "Depth positions:",
        len(d_positions)
    )

    print(
        "Height positions:",
        len(h_positions)
    )

    print(
        "Width positions:",
        len(w_positions)
    )

    print(
        "Total 3D patches:",
        total_patches
    )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    logits_sum = torch.zeros(
        (
            1,
            NUM_CLASSES,
            D,
            H,
            W
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    count_map = torch.zeros(
        (
            1,
            1,
            D,
            H,
            W
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    current = 0

    with torch.inference_mode():

        for d in d_positions:

            for h in h_positions:

                for w in w_positions:

                    current += 1

                    print(
                        f"\r3D Prediction: "
                        f"{current}/{total_patches}",
                        end="",
                        flush=True
                    )

                    patch = volume[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw
                    ]

                    patch = patch.to(
                        DEVICE
                    )

                    logits = model(
                        patch
                    )

                    logits = logits.float()

                    logits_sum[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw
                    ] += logits

                    count_map[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw
                    ] += 1.0

    print()

    # --------------------------------------------------------
    # Average overlapping predictions
    # --------------------------------------------------------

    logits_average = (
        logits_sum
        /
        count_map.clamp_min(1.0)
    )

    # --------------------------------------------------------
    # Convert logits to class labels
    # --------------------------------------------------------

    prediction = torch.argmax(
        logits_average,
        dim=1
    )

    prediction = prediction[
        0
    ].cpu().numpy()

    # --------------------------------------------------------
    # Remove padding
    # --------------------------------------------------------

    prediction = prediction[
        :depth,
        :height,
        :width
    ]

    prediction = prediction.astype(
        np.uint8
    )

    print()
    print(
        "3D prediction shape:",
        prediction.shape
    )

    print(
        "3D prediction completed."
    )

    return prediction


# ============================================================
# SAVE 3D NIFTI
# ============================================================

def save_segmentation_nifti(
    prediction,
    reference_nifti,
    output_path
):

    """
    Save complete 3D prediction as NIfTI.

    The affine and header from the original T1n
    are preserved so the segmentation stays in
    the same physical coordinate system.
    """

    print()
    print("=" * 70)
    print("SAVING 3D SEGMENTATION")
    print("=" * 70)

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Create NIfTI image
    # --------------------------------------------------------

    segmentation_nii = nib.Nifti1Image(
        prediction,
        reference_nifti.affine,
        reference_nifti.header
    )

    # Make datatype explicitly uint8
    segmentation_nii.set_data_dtype(
        np.uint8
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    nib.save(
        segmentation_nii,
        output_path
    )

    print()
    print(
        "3D segmentation saved:"
    )

    print(
        output_path
    )

    print()
    print(
        "Segmentation shape:",
        prediction.shape
    )

    print(
        "Voxel spacing:",
        reference_nifti.header.get_zooms()[:3]
    )


# ============================================================
# CHOOSE BEST SLICE
# ============================================================

def choose_slice(
    images,
    requested_slice=None
):

    volume = images["T1n"]

    depth = volume.shape[2]

    if requested_slice is not None:

        index = int(
            requested_slice
        )

        index = max(
            0,
            min(
                index,
                depth - 1
            )
        )

        return index

    nonzero_counts = []

    for z in range(depth):

        slice_data = volume[
            :,
            :,
            z
        ]

        count = np.count_nonzero(
            slice_data
        )

        nonzero_counts.append(
            count
        )

    index = int(
        np.argmax(
            nonzero_counts
        )
    )

    return index


# ============================================================
# DISPLAY NORMALIZATION
# ============================================================

def normalize_for_display(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    nonzero = image[
        image != 0
    ]

    if nonzero.size == 0:

        return np.zeros_like(
            image
        )

    low = np.percentile(
        nonzero,
        1
    )

    high = np.percentile(
        nonzero,
        99
    )

    if high <= low:

        return np.zeros_like(
            image
        )

    image = np.clip(
        image,
        low,
        high
    )

    image = (
        image - low
    ) / (
        high - low
    )

    return image


# ============================================================
# CREATE SEGMENTATION DISPLAY
# ============================================================

def create_segmentation_display(
    prediction
):

    prediction = prediction.astype(
        np.int32
    )

    display = np.zeros(
        prediction.shape,
        dtype=np.float32
    )

    display[
        prediction == 1
    ] = 1

    display[
        prediction == 2
    ] = 2

    display[
        prediction == 3
    ] = 3

    return display


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    images,
    prediction,
    slice_index,
    output_path
):

    print()
    print("=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Extract one slice ONLY for display
    # --------------------------------------------------------

    t1n = images["T1n"][
        :,
        :,
        slice_index
    ]

    t1c = images["T1c"][
        :,
        :,
        slice_index
    ]

    t2w = images["T2w"][
        :,
        :,
        slice_index
    ]

    t2f = images["T2f"][
        :,
        :,
        slice_index
    ]

    # Prediction is now 3D.
    # Only take one slice for the PNG.
    pred_slice = prediction[
        slice_index
    ]

    # --------------------------------------------------------
    # Normalize MRI
    # --------------------------------------------------------

    t1n = normalize_for_display(
        t1n
    )

    t1c = normalize_for_display(
        t1c
    )

    t2w = normalize_for_display(
        t2w
    )

    t2f = normalize_for_display(
        t2f
    )

    seg_display = create_segmentation_display(
        pred_slice
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 5)
    )

    # --------------------------------------------------------
    # T1n
    # --------------------------------------------------------

    axes[0].imshow(
        t1n.T,
        cmap="gray",
        origin="lower"
    )

    axes[0].set_title(
        "T1n",
        fontsize=14
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # T1c
    # --------------------------------------------------------

    axes[1].imshow(
        t1c.T,
        cmap="gray",
        origin="lower"
    )

    axes[1].set_title(
        "T1c",
        fontsize=14
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # T2w
    # --------------------------------------------------------

    axes[2].imshow(
        t2w.T,
        cmap="gray",
        origin="lower"
    )

    axes[2].set_title(
        "T2w",
        fontsize=14
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # T2f
    # --------------------------------------------------------

    axes[3].imshow(
        t2f.T,
        cmap="gray",
        origin="lower"
    )

    axes[3].set_title(
        "T2f",
        fontsize=14
    )

    axes[3].axis("off")

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    axes[4].imshow(
        seg_display.T,
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=3
    )

    axes[4].set_title(
        "Segmentation",
        fontsize=14
    )

    axes[4].axis("off")

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout(
        pad=2.0
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    print()
    print(
        "Visualization saved:"
    )

    print(
        output_path
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Full 3D Attention U-Net "
            "BraTS prediction"
        )
    )

    # --------------------------------------------------------
    # Patient directory
    # --------------------------------------------------------

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "Patient directory containing "
            "BraTS .nii.gz files"
        )
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Trained model checkpoint"
    )

    # --------------------------------------------------------
    # 3D NIfTI output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-nifti",
        type=str,
        default="results/segmentation_3d.nii.gz",
        help=(
            "Output 3D segmentation NIfTI"
        )
    )

    # --------------------------------------------------------
    # PNG visualization output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-image",
        type=str,
        default="results/prediction.png",
        help=(
            "Output visualization PNG"
        )
    )

    # --------------------------------------------------------
    # Display slice
    # --------------------------------------------------------

    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help=(
            "Axial slice index for visualization. "
            "If omitted, automatically chooses one."
        )
    )

    args = parser.parse_args()

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)
    print("3D ATTENTION U-NET PREDICTION")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Patient:",
        args.data_dir
    )

    print(
        "Checkpoint:",
        args.checkpoint
    )

    print(
        "3D NIfTI output:",
        args.output_nifti
    )

    print(
        "Visualization output:",
        args.output_image
    )

    print("=" * 70)

    # ========================================================
    # LOAD PATIENT
    # ========================================================

    images = load_patient(
        args.data_dir
    )

    # ========================================================
    # LOAD REFERENCE NIFTI
    # ========================================================

    reference_nifti = get_reference_nifti(
        args.data_dir
    )

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    ground_truth = load_ground_truth(
        args.data_dir
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = create_model()

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    model = load_checkpoint(
        model,
        args.checkpoint
    )

    # ========================================================
    # PREPARE INPUT
    # ========================================================

    volume = prepare_input(
        images
    )

    print()
    print(
        "Model input shape:",
        tuple(volume.shape)
    )

    # ========================================================
    # FULL 3D PREDICTION
    # ========================================================

    prediction = predict_volume(
        model=model,
        volume=volume
    )

    # ========================================================
    # PREDICTION INFORMATION
    # ========================================================

    unique_classes = np.unique(
        prediction
    )

    print()
    print(
        "Predicted classes:",
        unique_classes
    )

    for cls in unique_classes:

        count = np.sum(
            prediction == cls
        )

        print(
            f"Class {cls}: {count} voxels"
        )

    # ========================================================
    # SAVE 3D NIFTI
    # ========================================================

    save_segmentation_nifti(
        prediction=prediction,
        reference_nifti=reference_nifti,
        output_path=args.output_nifti
    )

    # ========================================================
    # CHOOSE VISUALIZATION SLICE
    # ========================================================

    slice_index = choose_slice(
        images,
        args.slice
    )

    print()
    print(
        "Visualization slice:",
        slice_index
    )

    # ========================================================
    # SAVE PNG VISUALIZATION
    # ========================================================

    save_visualization(
        images=images,
        prediction=prediction,
        slice_index=slice_index,
        output_path=args.output_image
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        "3D segmentation:"
    )

    print(
        args.output_nifti
    )

    print()
    print(
        "2D visualization:"
    )

    print(
        args.output_image
    )

    print()
    print(
        "Your friend can use the .nii.gz file "
        "for further 3D processing."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()