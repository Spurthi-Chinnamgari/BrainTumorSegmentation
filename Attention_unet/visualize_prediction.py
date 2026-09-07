"""
BraTS 2023 - 3D Attention U-Net Prediction
-------------------------------------------

Input:
    T1n
    T1c
    T2w
    T2f

Model:
    3D Attention U-Net

Classes produced by the trained model:
    0 = Background
    1 = NCR / NET
    2 = Edema
    3 = Enhancing Tumor

Segmentation visualization:
    Black  = Background
    Yellow = Tumor Core / NCR-NET
    Purple = Edema
    Red    = Enhancing Tumor

Derived BraTS regions:
    WT = NCR/NET + Edema + Enhancing Tumor
    TC = NCR/NET + Enhancing Tumor
    ET = Enhancing Tumor

Outputs:
    results/segmentation_3d.nii.gz
    results/prediction.png
"""

import os
import glob
import argparse

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn.functional as F

from models import AttentionUNet3D


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = (16, 32, 32)

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
# FIND MODALITY FILE
# ============================================================

def get_modality_file(patient_dir, modality):

    patient_dir = os.path.abspath(patient_dir)

    if not os.path.isdir(patient_dir):

        raise FileNotFoundError(
            "\nPatient directory does not exist:\n"
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
        "\nCould not find "
        f"*{modality}.nii.gz "
        "inside:\n"
        f"{patient_dir}\n"
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
        os.path.abspath(patient_dir)
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
            "\nMRI modalities do not have "
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
# REFERENCE NIFTI
# ============================================================

def get_reference_nifti(patient_dir):

    t1n_path = get_modality_file(
        patient_dir,
        "t1n"
    )

    return nib.load(t1n_path)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print()
    print("=" * 70)
    print("CREATING 3D ATTENTION U-NET")
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

    model = model.to(DEVICE)

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
    print("LOADING CHECKPOINT")
    print("=" * 70)

    checkpoint_path = os.path.abspath(
        checkpoint_path
    )

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
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

    if isinstance(checkpoint, dict):

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

    # Remove "module." if trained using DataParallel
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if len(missing) > 0:

        print(
            "\nWARNING: Missing keys:",
            len(missing)
        )

    if len(unexpected) > 0:

        print(
            "\nWARNING: Unexpected keys:",
            len(unexpected)
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

    volume = np.stack(
        [
            images["T1n"],
            images["T1c"],
            images["T2w"],
            images["T2f"],
        ],
        axis=0
    )

    tensor = torch.from_numpy(
        volume
    ).float()

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# GET PATCH POSITIONS
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

        positions.append(position)

        position += stride

    last_position = size - patch_size

    if (
        len(positions) == 0
        or positions[-1] != last_position
    ):

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

    print()
    print("=" * 70)
    print("FULL 3D PREDICTION")
    print("=" * 70)

    _, _, depth, height, width = volume.shape

    pd, ph, pw = PATCH_SIZE

    sd, sh, sw = STRIDE

    print(
        "Original volume:",
        (depth, height, width)
    )

    print(
        "Patch size:",
        PATCH_SIZE
    )

    print(
        "Stride:",
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

    _, _, D, H, W = volume.shape

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
        * len(h_positions)
        * len(w_positions)
    )

    print()
    print(
        "Total patches:",
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

    model.eval()

    with torch.inference_mode():

        for d in d_positions:

            for h in h_positions:

                for w in w_positions:

                    current += 1

                    print(
                        f"\rPredicting patch "
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
    # Convert to class labels
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

    return prediction


# ============================================================
# SAVE NIFTI
# ============================================================

def save_segmentation_nifti(
    prediction,
    reference_nifti,
    output_path
):

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

    segmentation_nii = nib.Nifti1Image(
        prediction,
        reference_nifti.affine,
        reference_nifti.header
    )

    segmentation_nii.set_data_dtype(
        np.uint8
    )

    nib.save(
        segmentation_nii,
        output_path
    )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# NORMALIZE MRI FOR DISPLAY
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

        return np.zeros_like(image)

    low = np.percentile(
        nonzero,
        1
    )

    high = np.percentile(
        nonzero,
        99
    )

    if high <= low:

        return np.zeros_like(image)

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
# CREATE COLORED SEGMENTATION MASK
# ============================================================

def create_colored_segmentation(
    prediction
):

    """
    EXACT COLORS:

    0 = Black      = Background
    1 = Yellow     = NCR/NET / Tumor Core
    2 = Purple     = Edema
    3 = Red        = Enhancing Tumor
    """

    # --------------------------------------------------------
    # Create custom colormap
    # --------------------------------------------------------

    cmap = ListedColormap(
        [
            "black",      # 0 Background
            "yellow",     # 1 NCR / NET
            "purple",     # 2 Edema
            "red",        # 3 Enhancing Tumor
        ]
    )

    return prediction, cmap


# ============================================================
# SAVE SEGMENTATION MASK ONLY
# ============================================================

def save_segmentation_mask(
    prediction,
    slice_index,
    output_path
):

    print()
    print("=" * 70)
    print("CREATING SEGMENTATION MASK")
    print("=" * 70)

    # --------------------------------------------------------
    # Take one axial slice
    # --------------------------------------------------------

    mask = prediction[
        :,
        :,
        slice_index
    ]

    mask, cmap = create_colored_segmentation(
        mask
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.imshow(
        mask.T,
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=3
    )

    ax.set_title(
        "Segmentation Mask",
        fontsize=18
    )

    ax.axis("off")

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    from matplotlib.patches import Patch

    legend_elements = [

        Patch(
            facecolor="red",
            label="Enhancing Tumor (ET)"
        ),

        Patch(
            facecolor="yellow",
            label="Tumor Core / NCR-NET (TC)"
        ),

        Patch(
            facecolor="purple",
            label="Edema"
        ),

        Patch(
            facecolor="black",
            edgecolor="white",
            label="Background"
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white"
    )

    print()
    print(
        "Segmentation mask saved:"
    )

    print(
        output_path
    )

    plt.show()

    plt.close()


# ============================================================
# CHOOSE BEST SLICE
# ============================================================

def choose_slice(
    images,
    prediction,
    requested_slice=None
):

    depth = prediction.shape[2]

    # User explicitly selected slice
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

    # --------------------------------------------------------
    # Prefer slice with maximum tumor pixels
    # --------------------------------------------------------

    tumor_counts = []

    for z in range(depth):

        mask = prediction[
            :,
            :,
            z
        ]

        tumor_pixels = np.count_nonzero(
            mask
        )

        tumor_counts.append(
            tumor_pixels
        )

    best_slice = int(
        np.argmax(
            tumor_counts
        )
    )

    # If no tumor was predicted,
    # use the middle slice.
    if tumor_counts[best_slice] == 0:

        best_slice = depth // 2

    return best_slice


# ============================================================
# PRINT CLASS INFORMATION
# ============================================================

def print_prediction_statistics(
    prediction
):

    print()
    print("=" * 70)
    print("PREDICTION STATISTICS")
    print("=" * 70)

    total_voxels = prediction.size

    for cls in range(NUM_CLASSES):

        count = np.sum(
            prediction == cls
        )

        percentage = (
            count /
            total_voxels
        ) * 100.0

        if cls == 0:
            name = "Background"

        elif cls == 1:
            name = "NCR / NET"

        elif cls == 2:
            name = "Edema"

        else:
            name = "Enhancing Tumor"

        print(
            f"Class {cls} "
            f"({name:20s}): "
            f"{count:10d} voxels "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Derived BraTS regions
    # --------------------------------------------------------

    wt = (
        prediction > 0
    )

    tc = (
        (prediction == 1)
        |
        (prediction == 3)
    )

    et = (
        prediction == 3
    )

    print()
    print(
        "Derived BraTS regions:"
    )

    print(
        "Whole Tumor (WT):",
        np.sum(wt),
        "voxels"
    )

    print(
        "Tumor Core (TC):",
        np.sum(tc),
        "voxels"
    )

    print(
        "Enhancing Tumor (ET):",
        np.sum(et),
        "voxels"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "3D Attention U-Net "
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
            "Path to ONE patient directory "
            "containing t1n/t1c/t2w/t2f files"
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
    # NIFTI output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-nifti",
        type=str,
        default="results/segmentation_3d.nii.gz"
    )

    # --------------------------------------------------------
    # Mask output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-mask",
        type=str,
        default="results/segmentation_mask.png"
    )

    # --------------------------------------------------------
    # Slice
    # --------------------------------------------------------

    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help=(
            "Axial slice index. "
            "If omitted, slice with maximum "
            "predicted tumor is selected."
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
        "Output NIfTI:",
        args.output_nifti
    )

    print(
        "Output Mask:",
        args.output_mask
    )

    print("=" * 70)

    # ========================================================
    # LOAD PATIENT
    # ========================================================

    images = load_patient(
        args.data_dir
    )

    # ========================================================
    # REFERENCE
    # ========================================================

    reference_nifti = get_reference_nifti(
        args.data_dir
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model()

    # ========================================================
    # CHECKPOINT
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

    print()
    print(
        "Prediction shape:",
        prediction.shape
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print_prediction_statistics(
        prediction
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
    # CHOOSE SLICE
    # ========================================================

    slice_index = choose_slice(
        images,
        prediction,
        args.slice
    )

    print()
    print(
        "Selected visualization slice:",
        slice_index
    )

    # ========================================================
    # SAVE MASK ONLY
    # ========================================================

    save_segmentation_mask(
        prediction=prediction,
        slice_index=slice_index,
        output_path=args.output_mask
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
        os.path.abspath(
            args.output_nifti
        )
    )

    print()
    print(
        "Segmentation mask:"
    )

    print(
        os.path.abspath(
            args.output_mask
        )
    )

    print()
    print(
        "The PNG contains ONLY the segmentation mask."
    )

    print(
        "Background = Black"
    )

    print(
        "NCR/NET / Tumor Core = Yellow"
    )

    print(
        "Edema = Purple"
    )

    print(
        "Enhancing Tumor = Red"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()