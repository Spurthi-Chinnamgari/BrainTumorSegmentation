import os
import sys
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from Attention_unet.models.attention_unet_3d import AttentionUNet3D


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ATTENTION_UNET_DIR = os.path.join(
    PROJECT_ROOT,
    "Attention_unet"
)

PROCESSED_ROOT = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "data",
    "processed"
)

SPLIT_ROOT = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "data",
    "splits"
)

CHECKPOINT = os.path.join(
    ATTENTION_UNET_DIR,
    "checkpoints",
    "best_model.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "postprocessing",
    "gradcam_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

PATIENT_ID = None
PATCH_INDEX = 0
TARGET_CLASS = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FIND PATIENT
# ============================================================

def get_patient_id():

    if PATIENT_ID is not None:
        return PATIENT_ID

    test_file = os.path.join(
        SPLIT_ROOT,
        "test.txt"
    )

    with open(test_file, "r") as f:
        patients = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if len(patients) == 0:
        raise RuntimeError("No patient IDs found in test.txt")

    return patients[0]


# ============================================================
# LOAD MODEL
# ============================================================

print("Device:", DEVICE)

model = AttentionUNet3D(
    in_channels=4,
    out_channels=4,
    features=[16, 32, 64, 128, 256],
    dropout=0.1,
    use_transpose=True
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model.eval()

print("Model loaded successfully")


# ============================================================
# LOAD INPUT
# ============================================================

patient_id = get_patient_id()

print("Patient:", patient_id)

images_path = os.path.join(
    PROCESSED_ROOT,
    "test",
    patient_id,
    "images.npy"
)

if not os.path.exists(images_path):
    raise FileNotFoundError(
        f"Input file not found:\n{images_path}"
    )

images = np.load(images_path)

print("Images shape:", images.shape)

if images.ndim != 5:
    raise ValueError(
        "Expected images.npy shape: (N, 4, D, H, W)"
    )

if PATCH_INDEX >= images.shape[0]:
    raise IndexError(
        f"PATCH_INDEX {PATCH_INDEX} is out of range. "
        f"Available patches: {images.shape[0]}"
    )

patch = images[PATCH_INDEX]

patch_tensor = torch.tensor(
    patch,
    dtype=torch.float32,
    device=DEVICE
).unsqueeze(0)

print("Patch shape:", patch_tensor.shape)


# ============================================================
# GRAD-CAM HOOKS
# ============================================================

activations = []
gradients = []

target_layer = model.decoder_blocks[-1].conv


def forward_hook(module, input, output):
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# FORWARD PASS
# ============================================================

model.zero_grad()

output = model(patch_tensor)

print("Model output shape:", output.shape)


# ============================================================
# TARGET CLASS
# ============================================================

class_output = output[:, TARGET_CLASS, :, :, :]

prediction = torch.argmax(
    output,
    dim=1
)

target_voxels = (
    prediction == TARGET_CLASS
)

if target_voxels.sum() == 0:
    print("Target class not found in prediction.")
    print("No Grad-CAM heatmap generated.")

    forward_handle.remove()
    backward_handle.remove()

    sys.exit()


score = class_output[target_voxels].mean()

print("Target class:", TARGET_CLASS)
print("Target voxels:", target_voxels.sum().item())

score.backward()


# ============================================================
# GRAD-CAM
# ============================================================

activation = activations[0]
gradient = gradients[0]

weights = gradient.mean(
    dim=(2, 3, 4),
    keepdim=True
)

cam = (
    weights * activation
).sum(dim=1, keepdim=True)

cam = F.relu(cam)

cam = F.interpolate(
    cam,
    size=patch_tensor.shape[2:],
    mode="trilinear",
    align_corners=False
)

cam = cam.squeeze().detach().cpu().numpy()


# ============================================================
# NORMALIZE
# ============================================================

cam_min = cam.min()
cam_max = cam.max()

if cam_max > cam_min:
    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min
    )
else:
    cam = np.zeros_like(cam)


# ============================================================
# SAVE NPY
# ============================================================

npy_path = os.path.join(
    OUTPUT_DIR,
    f"{patient_id}_patch{PATCH_INDEX}_class{TARGET_CLASS}_gradcam.npy"
)

np.save(
    npy_path,
    cam
)


# ============================================================
# SAVE NIFTI
# ============================================================

nii_path = os.path.join(
    OUTPUT_DIR,
    f"{patient_id}_patch{PATCH_INDEX}_class{TARGET_CLASS}_gradcam.nii.gz"
)

nii_img = nib.Nifti1Image(
    cam.astype(np.float32),
    np.eye(4)
)

nib.save(
    nii_img,
    nii_path
)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

middle_slice = cam.shape[0] // 2

plt.figure(figsize=(6, 6))

plt.imshow(
    cam[middle_slice],
    cmap="jet"
)

plt.colorbar(
    label="Grad-CAM intensity"
)

plt.title(
    f"3D Grad-CAM\nPatient: {patient_id}\nClass: {TARGET_CLASS}"
)

plt.axis("off")

png_path = os.path.join(
    OUTPUT_DIR,
    f"{patient_id}_patch{PATCH_INDEX}_class{TARGET_CLASS}_gradcam.png"
)

plt.savefig(
    png_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CLEANUP
# ============================================================

forward_handle.remove()
backward_handle.remove()


print()
print("Grad-CAM completed successfully")
print("Saved NPY :", npy_path)
print("Saved NIfTI:", nii_path)
print("Saved PNG :", png_path)