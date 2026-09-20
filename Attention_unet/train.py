"""
3D Attention U-Net Training Pipeline
------------------------------------

Uses the VERIFIED preprocessing output.

Pipeline:

Raw BraTS 2023
      ↓
Verified preprocessing
      ↓
Normalization + cropping
      ↓
64 x 64 x 64 patches
      ↓
images.npy / masks.npy
      ↓
Exact train / val / test split
      ↓
3D Attention U-Net
      ↓
4-class segmentation

Classes:
    0 = Background
    1 = NCR / NET
    2 = Edema
    3 = Enhancing Tumor

IMPORTANT:
    This file does NOT perform:
        - Z-score normalization
        - random cropping
        - random patient splitting

Those steps are already handled by preprocessing.
"""


import os
import time
import argparse

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import get_default_config

from dataset import (
    BraTSDataset3D,
    read_split_file,
)

from models import AttentionUNet3D

from losses import DiceBCELoss

from postprocessing.metrics import (
    compute_dice_score,
    compute_iou_score,
    compute_brats_regions_metrics,
)

from utils import (
    setup_logger,
    MetricTracker,
    TensorBoardLogger,
    CheckpointManager,
    save_prediction_comparison,
)


# ============================================================
# CONSTANTS
# ============================================================

PATCH_SIZE = (64, 64, 64)

NUM_CLASSES = 4
NUM_MODALITIES = 4

# Exact verified split
EXPECTED_TRAIN_PATIENTS = 26
EXPECTED_VAL_PATIENTS = 6
EXPECTED_TEST_PATIENTS = 5

# Validate every epoch because validation is now patch based
VALIDATE_EVERY = 1


# ============================================================
# FIND PROCESSED DATA
# ============================================================

def find_data_paths(
    processed_root=None,
    split_root=None,
):
    """
    Automatically locate the preprocessing output.

    Possible structures supported:

        BrainTumorSegmentation/
        ├── preprocessing/
        │   └── data/
        │       ├── processed/
        │       └── splits/
        │
        └── Attention_unet/

    OR:

        BrainTumorSegmentation/
        ├── data/
        │   ├── processed/
        │   └── splits/
        │
        └── Attention_unet/
    """

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.dirname(
        current_dir
    )

    # --------------------------------------------------------
    # User supplied paths have highest priority
    # --------------------------------------------------------

    if processed_root is not None:
        processed_root = os.path.abspath(
            processed_root
        )

    if split_root is not None:
        split_root = os.path.abspath(
            split_root
        )

    # --------------------------------------------------------
    # Candidate locations
    # --------------------------------------------------------

    processed_candidates = [
        os.path.join(
            project_root,
            "preprocessing",
            "data",
            "processed",
        ),

        os.path.join(
            project_root,
            "data",
            "processed",
        ),

        os.path.join(
            current_dir,
            "data",
            "processed",
        ),

        os.path.join(
            os.getcwd(),
            "preprocessing",
            "data",
            "processed",
        ),

        os.path.join(
            os.getcwd(),
            "data",
            "processed",
        ),
    ]

    split_candidates = [
        os.path.join(
            project_root,
            "preprocessing",
            "data",
            "splits",
        ),

        os.path.join(
            project_root,
            "data",
            "splits",
        ),

        os.path.join(
            current_dir,
            "data",
            "splits",
        ),

        os.path.join(
            os.getcwd(),
            "preprocessing",
            "data",
            "splits",
        ),

        os.path.join(
            os.getcwd(),
            "data",
            "splits",
        ),
    ]

    # --------------------------------------------------------
    # Resolve processed root
    # --------------------------------------------------------

    if processed_root is None:

        for path in processed_candidates:

            if os.path.isdir(path):

                processed_root = path
                break

    # --------------------------------------------------------
    # Resolve split root
    # --------------------------------------------------------

    if split_root is None:

        for path in split_candidates:

            if os.path.isdir(path):

                split_root = path
                break

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if processed_root is None:

        raise FileNotFoundError(
            "\nCould not find processed data.\n\n"
            "Expected one of:\n"
            + "\n".join(processed_candidates)
            + "\n\n"
            "Use --processed-root to specify it manually."
        )

    if split_root is None:

        raise FileNotFoundError(
            "\nCould not find split files.\n\n"
            "Expected one of:\n"
            + "\n".join(split_candidates)
            + "\n\n"
            "Use --split-root to specify it manually."
        )

    return (
        os.path.abspath(processed_root),
        os.path.abspath(split_root),
    )


# ============================================================
# VERIFY SPLITS
# ============================================================

def verify_splits(
    split_root,
):
    """
    Verify the exact preprocessing split.

    Expected:
        train = 26
        val   = 6
        test  = 5
    """

    train_file = os.path.join(
        split_root,
        "train.txt",
    )

    val_file = os.path.join(
        split_root,
        "val.txt",
    )

    test_file = os.path.join(
        split_root,
        "test.txt",
    )

    train_ids = read_split_file(
        train_file
    )

    val_ids = read_split_file(
        val_file
    )

    test_ids = read_split_file(
        test_file
    )

    # --------------------------------------------------------
    # Check counts
    # --------------------------------------------------------

    if len(train_ids) != EXPECTED_TRAIN_PATIENTS:

        raise ValueError(
            "\nTrain split mismatch.\n"
            f"Expected: {EXPECTED_TRAIN_PATIENTS}\n"
            f"Found:    {len(train_ids)}\n"
            f"File:     {train_file}"
        )

    if len(val_ids) != EXPECTED_VAL_PATIENTS:

        raise ValueError(
            "\nValidation split mismatch.\n"
            f"Expected: {EXPECTED_VAL_PATIENTS}\n"
            f"Found:    {len(val_ids)}\n"
            f"File:     {val_file}"
        )

    if len(test_ids) != EXPECTED_TEST_PATIENTS:

        raise ValueError(
            "\nTest split mismatch.\n"
            f"Expected: {EXPECTED_TEST_PATIENTS}\n"
            f"Found:    {len(test_ids)}\n"
            f"File:     {test_file}"
        )

    # --------------------------------------------------------
    # Check for overlap
    # --------------------------------------------------------

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    if train_set & val_set:

        raise ValueError(
            "Train and validation sets overlap."
        )

    if train_set & test_set:

        raise ValueError(
            "Train and test sets overlap."
        )

    if val_set & test_set:

        raise ValueError(
            "Validation and test sets overlap."
        )

    print()
    print("=" * 70)
    print("EXACT PREPROCESSING SPLIT")
    print("=" * 70)

    print(
        f"Train patients : {len(train_ids)}"
    )

    print(
        f"Val patients   : {len(val_ids)}"
    )

    print(
        f"Test patients  : {len(test_ids)}"
    )

    print(
        "No patient overlap detected."
    )

    print("=" * 70)

    return (
        train_ids,
        val_ids,
        test_ids,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
):

    model.train()

    loss_tracker = MetricTracker()

    dice_tracker = MetricTracker()

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [Train]",
    )

    for batch in pbar:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        # ====================================================
        # OPTIONAL PATCH AUGMENTATION
        #
        # Preprocessing has already normalized and cropped.
        # We therefore DO NOT normalize or crop again.
        #
        # Random spatial flips are safe to apply here.
        # ====================================================

        if torch.rand(1).item() < 0.5:

            images = torch.flip(
                images,
                dims=[2],
            )

            masks = torch.flip(
                masks,
                dims=[1],
            )

        if torch.rand(1).item() < 0.5:

            images = torch.flip(
                images,
                dims=[3],
            )

            masks = torch.flip(
                masks,
                dims=[2],
            )

        if torch.rand(1).item() < 0.5:

            images = torch.flip(
                images,
                dims=[4],
            )

            masks = torch.flip(
                masks,
                dims=[3],
            )

        # ====================================================
        # ZERO GRADIENT
        # ====================================================

        optimizer.zero_grad(
            set_to_none=True
        )

        # ====================================================
        # FORWARD + LOSS
        # ====================================================

        if scaler is not None:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    masks,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks,
            )

            loss.backward()

            optimizer.step()

        # ====================================================
        # OUTPUT SHAPE CHECK
        # ====================================================

        if logits.ndim != 5:

            raise RuntimeError(
                "\nUnexpected model output.\n"
                f"Expected: (B, 4, 64, 64, 64)\n"
                f"Got:      {tuple(logits.shape)}"
            )

        if logits.shape[1] != NUM_CLASSES:

            raise RuntimeError(
                "\nUnexpected number of output classes.\n"
                f"Expected: {NUM_CLASSES}\n"
                f"Got:      {logits.shape[1]}"
            )

        # ====================================================
        # DICE
        # ====================================================

        with torch.no_grad():

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            mean_dice, _ = compute_dice_score(
                predictions,
                masks,
            )

        batch_size = images.size(0)

        loss_tracker.update(
            loss.item(),
            batch_size,
        )

        dice_tracker.update(
            mean_dice,
            batch_size,
        )

        pbar.set_postfix(
            {
                "Loss":
                    f"{loss_tracker.avg:.4f}",

                "Dice":
                    f"{dice_tracker.avg:.4f}",
            }
        )

    return {
        "loss": loss_tracker.avg,
        "dice": dice_tracker.avg,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    dataloader,
    criterion,
    device,
    epoch,
):

    model.eval()

    loss_tracker = MetricTracker()

    dice_tracker = MetricTracker()

    iou_tracker = MetricTracker()

    wt_tracker = MetricTracker()

    tc_tracker = MetricTracker()

    et_tracker = MetricTracker()

    sample_for_viz = None

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [Val]",
    )

    with torch.no_grad():

        for batch_index, batch in enumerate(
            pbar
        ):

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            # =================================================
            # FORWARD
            # =================================================

            if device.type == "cuda":

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    logits = model(
                        images
                    )

                    loss = criterion(
                        logits,
                        masks,
                    )

            else:

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    masks,
                )

            # =================================================
            # PREDICTION
            # =================================================

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            # =================================================
            # METRICS
            # =================================================

            mean_dice, _ = compute_dice_score(
                predictions,
                masks,
            )

            mean_iou, _ = compute_iou_score(
                predictions,
                masks,
            )

            brats_metrics = (
                compute_brats_regions_metrics(
                    predictions,
                    masks,
                )
            )

            batch_size = images.size(0)

            loss_tracker.update(
                loss.item(),
                batch_size,
            )

            dice_tracker.update(
                mean_dice,
                batch_size,
            )

            iou_tracker.update(
                mean_iou,
                batch_size,
            )

            wt_tracker.update(
                brats_metrics["dice_wt"],
                batch_size,
            )

            tc_tracker.update(
                brats_metrics["dice_tc"],
                batch_size,
            )

            et_tracker.update(
                brats_metrics["dice_et"],
                batch_size,
            )

            # =================================================
            # FIRST VALIDATION SAMPLE
            # =================================================

            if batch_index == 0:

                sample_for_viz = (
                    images[0]
                    .detach()
                    .cpu()
                    .numpy(),

                    masks[0]
                    .detach()
                    .cpu()
                    .numpy(),

                    predictions[0]
                    .detach()
                    .cpu()
                    .numpy(),
                )

            pbar.set_postfix(
                {
                    "Loss":
                        f"{loss_tracker.avg:.4f}",

                    "Dice":
                        f"{dice_tracker.avg:.4f}",

                    "IoU":
                        f"{iou_tracker.avg:.4f}",

                    "WT":
                        f"{wt_tracker.avg:.4f}",

                    "TC":
                        f"{tc_tracker.avg:.4f}",

                    "ET":
                        f"{et_tracker.avg:.4f}",
                }
            )

    return {
        "val_loss":
            loss_tracker.avg,

        "val_dice":
            dice_tracker.avg,

        "val_iou":
            iou_tracker.avg,

        "val_dice_wt":
            wt_tracker.avg,

        "val_dice_tc":
            tc_tracker.avg,

        "val_dice_et":
            et_tracker.avg,

        "sample_viz":
            sample_for_viz,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train 3D Attention U-Net "
            "using verified processed BraTS data."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--processed-root",
        type=str,
        default=None,
        help=(
            "Path to data/processed"
        ),
    )

    parser.add_argument(
        "--split-root",
        type=str,
        default=None,
        help=(
            "Path to data/splits"
        ),
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    # ========================================================
    # CONFIG
    # ========================================================

    cfg = get_default_config()

    cfg.train.dimension = "3d"

    cfg.train.epochs = args.epochs

    cfg.train.batch_size = args.batch_size

    cfg.train.learning_rate = args.lr

    cfg.train.early_stopping_patience = (
        args.patience
    )

    # ========================================================
    # FIND PROCESSED DATA
    # ========================================================

    processed_root, split_root = (
        find_data_paths(
            processed_root=args.processed_root,
            split_root=args.split_root,
        )
    )

    # ========================================================
    # VERIFY SPLIT
    # ========================================================

    (
        train_ids,
        val_ids,
        test_ids,
    ) = verify_splits(
        split_root
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)
    print("3D ATTENTION U-NET TRAINING")
    print("=" * 70)

    print(
        "Data source       : VERIFIED PREPROCESSED DATA"
    )

    print(
        "Input modalities  : 4"
    )

    print(
        "Patch size        :",
        PATCH_SIZE,
    )

    print(
        "Output classes    :",
        NUM_CLASSES,
    )

    print(
        "Train patients    :",
        len(train_ids),
    )

    print(
        "Validation patients:",
        len(val_ids),
    )

    print(
        "Test patients     :",
        len(test_ids),
    )

    print(
        "Device            :",
        device,
    )

    print(
        "Processed root    :",
        processed_root,
    )

    print(
        "Split root        :",
        split_root,
    )

    print("=" * 70)

    # ========================================================
    # LOGGER
    # ========================================================

    logger = setup_logger(
        log_file=os.path.join(
            cfg.train.log_dir,
            "train.log",
        )
    )

    logger.info(
        "Starting Attention U-Net training "
        "using verified processed BraTS data."
    )

    # ========================================================
    # DATASETS
    # ========================================================

    print()
    print(
        "Loading processed training dataset..."
    )

    train_ds = BraTSDataset3D(
        processed_root=processed_root,
        split="train",
        split_file=os.path.join(
            split_root,
            "train.txt",
        ),
        patch_size=PATCH_SIZE,
    )

    print()
    print(
        "Loading processed validation dataset..."
    )

    val_ds = BraTSDataset3D(
        processed_root=processed_root,
        split="val",
        split_file=os.path.join(
            split_root,
            "val.txt",
        ),
        patch_size=PATCH_SIZE,
    )

    # ========================================================
    # DATASET COUNTS
    # ========================================================

    print()
    print("=" * 70)

    print(
        "PROCESSED DATASET"
    )

    print("=" * 70)

    print(
        f"Train patients : {len(train_ids)}"
    )

    print(
        f"Train patches  : {len(train_ds)}"
    )

    print(
        f"Val patients   : {len(val_ids)}"
    )

    print(
        f"Val patches     : {len(val_ds)}"
    )

    print(
        f"Test patients  : {len(test_ids)}"
    )

    print("=" * 70)

    # ========================================================
    # VERIFY ONE SAMPLE
    # ========================================================

    sample = train_ds[0]

    sample_image = sample["image"]
    sample_mask = sample["mask"]

    print()
    print(
        "FIRST PROCESSED SAMPLE"
    )

    print(
        "Image shape:",
        tuple(sample_image.shape),
    )

    print(
        "Mask shape :",
        tuple(sample_mask.shape),
    )

    print(
        "Image dtype:",
        sample_image.dtype,
    )

    print(
        "Mask dtype :",
        sample_mask.dtype,
    )

    print(
        "Patient    :",
        sample["patient_id"],
    )

    print(
        "Patch      :",
        sample["patch_index"],
    )

    expected_image_shape = (
        NUM_MODALITIES,
        *PATCH_SIZE
    )

    if tuple(sample_image.shape) != (
        expected_image_shape
    ):

        raise RuntimeError(
            "\nProcessed image shape is wrong.\n"
            f"Expected: {expected_image_shape}\n"
            f"Got:      {tuple(sample_image.shape)}"
        )

    if tuple(sample_mask.shape) != (
        PATCH_SIZE
    ):

        raise RuntimeError(
            "\nProcessed mask shape is wrong.\n"
            f"Expected: {PATCH_SIZE}\n"
            f"Got:      {tuple(sample_mask.shape)}"
        )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # ========================================================
    # VERIFY BATCH SHAPE
    # ========================================================

    first_batch = next(
        iter(train_loader)
    )

    print()
    print(
        "FIRST TRAIN BATCH"
    )

    print(
        "Image batch:",
        tuple(
            first_batch["image"].shape
        ),
    )

    print(
        "Mask batch :",
        tuple(
            first_batch["mask"].shape
        ),
    )

    expected_batch_shape = (
        cfg.train.batch_size,
        NUM_MODALITIES,
        *PATCH_SIZE
    )

    if tuple(
        first_batch["image"].shape
    ) != expected_batch_shape:

        raise RuntimeError(
            "\nIncorrect batch entering model.\n"
            f"Expected: {expected_batch_shape}\n"
            f"Got:      "
            f"{tuple(first_batch['image'].shape)}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Creating Attention U-Net 3D..."
    )

    model = AttentionUNet3D(
        in_channels=NUM_MODALITIES,
        out_channels=NUM_CLASSES,

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
    # MODEL OUTPUT VERIFICATION
    # ========================================================

    print(
        "Checking model input/output shapes..."
    )

    model.eval()

    with torch.no_grad():

        test_input = first_batch[
            "image"
        ][:1].to(device)

        test_output = model(
            test_input
        )

    expected_output_shape = (
        1,
        NUM_CLASSES,
        *PATCH_SIZE
    )

    print(
        "Model input :",
        tuple(test_input.shape),
    )

    print(
        "Model output:",
        tuple(test_output.shape),
    )

    if tuple(
        test_output.shape
    ) != expected_output_shape:

        raise RuntimeError(
            "\nUnexpected Attention U-Net output.\n"
            f"Expected: {expected_output_shape}\n"
            f"Got:      {tuple(test_output.shape)}"
        )

    test_prediction = torch.argmax(
        test_output,
        dim=1,
    )

    print(
        "Argmax output:",
        tuple(
            test_prediction.shape
        ),
    )

    print(
        "Initial prediction labels:",
        torch.unique(
            test_prediction
        ).detach().cpu().tolist(),
    )

    del test_input
    del test_output
    del test_prediction

    if device.type == "cuda":

        torch.cuda.empty_cache()

    model.train()

    # ========================================================
    # LOSS
    # ========================================================

    criterion = DiceBCELoss(
        dice_weight=cfg.train.dice_weight,
        bce_weight=cfg.train.bce_weight,
    ).to(device)

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.AdamW(
        model.parameters(),

        lr=cfg.train.learning_rate,

        weight_decay=cfg.train.weight_decay,
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,

            T_max=cfg.train.epochs,

            eta_min=cfg.train.min_lr,
        )
    )

    # ========================================================
    # AMP
    # ========================================================

    scaler = None

    if (
        device.type == "cuda"
        and cfg.train.use_amp
        and not args.no_amp
    ):

        scaler = torch.amp.GradScaler(
            "cuda"
        )

        print(
            "AMP: ENABLED"
        )

    else:

        print(
            "AMP: DISABLED"
        )

    # ========================================================
    # LOGGING
    # ========================================================

    tb_logger = TensorBoardLogger(
        log_dir=cfg.train.log_dir
    )

    ckpt_manager = CheckpointManager(
        checkpoint_dir=cfg.train.checkpoint_dir,

        metric_name="val_dice",

        mode="max",
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_val_dice = 0.0

    patience_counter = 0

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        cfg.train.epochs + 1,
    ):

        start_time = time.time()

        # ====================================================
        # TRAIN
        # ====================================================

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step()

        elapsed = (
            time.time()
            - start_time
        )

        # ====================================================
        # PRINT
        # ====================================================

        print()
        print("-" * 70)

        print(
            f"Epoch {epoch:03d}/"
            f"{cfg.train.epochs:03d}"
        )

        print(
            f"Train Loss : "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Train Dice : "
            f"{train_metrics['dice']:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_metrics['val_loss']:.4f}"
        )

        print(
            f"Val Dice   : "
            f"{val_metrics['val_dice']:.4f}"
        )

        print(
            f"Val IoU    : "
            f"{val_metrics['val_iou']:.4f}"
        )

        print(
            f"WT Dice    : "
            f"{val_metrics['val_dice_wt']:.4f}"
        )

        print(
            f"TC Dice    : "
            f"{val_metrics['val_dice_tc']:.4f}"
        )

        print(
            f"ET Dice    : "
            f"{val_metrics['val_dice_et']:.4f}"
        )

        print(
            f"Learning rate: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

        print(
            f"Epoch time: "
            f"{elapsed / 60:.2f} minutes"
        )

        print("-" * 70)

        # ====================================================
        # LOGGER
        # ====================================================

        logger.info(
            f"Epoch [{epoch:03d}/"
            f"{cfg.train.epochs:03d}] | "

            f"Train Loss: "
            f"{train_metrics['loss']:.4f} | "

            f"Train Dice: "
            f"{train_metrics['dice']:.4f} | "

            f"Val Loss: "
            f"{val_metrics['val_loss']:.4f} | "

            f"Val Dice: "
            f"{val_metrics['val_dice']:.4f} | "

            f"Val IoU: "
            f"{val_metrics['val_iou']:.4f}"
        )

        # ====================================================
        # TENSORBOARD
        # ====================================================

        tb_logger.log_scalar(
            "Loss/Train",
            train_metrics["loss"],
            epoch,
        )

        tb_logger.log_scalar(
            "Dice/Train",
            train_metrics["dice"],
            epoch,
        )

        tb_logger.log_scalar(
            "Loss/Val",
            val_metrics["val_loss"],
            epoch,
        )

        tb_logger.log_scalar(
            "Dice/Val",
            val_metrics["val_dice"],
            epoch,
        )

        tb_logger.log_scalar(
            "IoU/Val",
            val_metrics["val_iou"],
            epoch,
        )

        tb_logger.log_scalar(
            "Dice/Val_WT",
            val_metrics["val_dice_wt"],
            epoch,
        )

        tb_logger.log_scalar(
            "Dice/Val_TC",
            val_metrics["val_dice_tc"],
            epoch,
        )

        tb_logger.log_scalar(
            "Dice/Val_ET",
            val_metrics["val_dice_et"],
            epoch,
        )

        tb_logger.log_scalar(
            "LearningRate",
            optimizer.param_groups[0]["lr"],
            epoch,
        )

        # ====================================================
        # CHECKPOINT
        # ====================================================

        is_best = ckpt_manager.step(
            current_metric=val_metrics["val_dice"],

            epoch=epoch,

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        if is_best:

            best_val_dice = (
                val_metrics["val_dice"]
            )

            patience_counter = 0

            logger.info(
                f"[BEST MODEL] "
                f"Validation Dice = "
                f"{best_val_dice:.4f}"
            )

            print()
            print(
                "BEST MODEL SAVED"
            )

            print(
                f"Best Validation Dice: "
                f"{best_val_dice:.4f}"
            )

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            if (
                val_metrics["sample_viz"]
                is not None
            ):

                img, tgt, prd = (
                    val_metrics[
                        "sample_viz"
                    ]
                )

                viz_path = os.path.join(
                    cfg.train.results_dir,

                    f"best_val_epoch_{epoch}.png",
                )

                try:

                    save_prediction_comparison(
                        img,
                        tgt,
                        prd,
                        save_path=viz_path,
                        title=(
                            f"Epoch {epoch} "
                            f"Best Validation"
                        ),
                    )

                    print(
                        "Saved visualization:",
                        viz_path,
                    )

                except Exception as e:

                    logger.warning(
                        f"Could not save "
                        f"visualization: {e}"
                    )

        else:

            patience_counter += 1

            print(
                f"No improvement. "
                f"Patience: "
                f"{patience_counter}/"
                f"{cfg.train.early_stopping_patience}"
            )

            if (
                patience_counter
                >=
                cfg.train.early_stopping_patience
            ):

                logger.info(
                    "Early stopping triggered."
                )

                print()
                print(
                    "Early stopping triggered."
                )

                break

    # ========================================================
    # FINISH
    # ========================================================

    tb_logger.close()

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best Validation Dice: "
        f"{best_val_dice:.4f}"
    )

    print(
        "Train patients:",
        len(train_ids),
    )

    print(
        "Validation patients:",
        len(val_ids),
    )

    print(
        "Test patients:",
        len(test_ids),
    )

    print(
        "Checkpoint directory:",
        cfg.train.checkpoint_dir,
    )

    print(
        "Results directory:",
        cfg.train.results_dir,
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()