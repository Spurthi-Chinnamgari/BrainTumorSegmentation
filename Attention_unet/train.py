"""
Fast 3D Training Pipeline
-------------------------

3D Attention U-Net for BraTS 2023 multimodal brain tumor
segmentation.

REAL BraTS DATA ONLY.

Input modalities:
    1. T1 native  -> t1n
    2. T1ce       -> t1c
    3. T2         -> t2w
    4. FLAIR      -> t2f

Ground Truth:
    *_seg.nii.gz

Classes:
    0 = Background
    1 = NCR / NET
    2 = Edema
    3 = Enhancing Tumor

Training:
    100 real BraTS patients

Validation:
    20 real BraTS patients

Training patch:
    (16, 32, 32)

Validation:
    Full 3D volume using fast sliding-window inference.

IMPORTANT:
    This file does NOT generate or use synthetic data.
"""


import os
import time
import argparse

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import get_default_config

from dataset import (
    BraTSDataset3D,
    create_train_val_test_split,
)

from dataset.transforms import (
    ComposeTransforms,
    ZScoreNormalize,
    RandomCrop3D,
    RandomFlip3D,
    ToTensor,
)

from models import AttentionUNet3D

from losses import DiceBCELoss

from metrics import (
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
# SETTINGS
# ============================================================

PATCH_SIZE = (16, 32, 32)

# ============================================================
# FAST VALIDATION
#
# Same patch size, but NO overlap.
#
# Old:
# STRIDE = (8, 16, 16)
#
# New:
# STRIDE = PATCH_SIZE
#
# This greatly reduces the number of validation patches.
# ============================================================

STRIDE = PATCH_SIZE

NUM_CLASSES = 4

NUM_MODALITIES = 4

MAX_TRAIN_PATIENTS = 100

MAX_VAL_PATIENTS = 20

# Validate every 5 epochs.
# Set to 1 if you want validation after every epoch.
VALIDATE_EVERY = 5


# ============================================================
# PATCH POSITION GENERATOR
# ============================================================

def get_positions(size, patch, stride):
    """
    Generate sliding-window starting positions.

    The final position is always included so that the complete
    volume is covered.
    """

    if size <= patch:
        return [0]

    positions = []

    pos = 0

    while pos + patch < size:

        positions.append(pos)

        pos += stride

    last = size - patch

    if len(positions) == 0 or positions[-1] != last:
        positions.append(last)

    return positions


# ============================================================
# PAD VOLUME
# ============================================================

def pad_volume(image, patch_size):
    """
    Pad image only when smaller than patch size.

    image:
        (B, C, D, H, W)
    """

    _, _, d, h, w = image.shape

    pd = max(0, patch_size[0] - d)
    ph = max(0, patch_size[1] - h)
    pw = max(0, patch_size[2] - w)

    if pd == 0 and ph == 0 and pw == 0:
        return image, (0, 0, 0)

    image = torch.nn.functional.pad(
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
# FAST 3D SLIDING-WINDOW PREDICTION
# ============================================================

def sliding_window_prediction(
    model,
    image,
    device,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
):
    """
    Predict a complete 3D MRI volume.

    Input:
        image = (1, 4, D, H, W)

    Output:
        prediction = (D, H, W)

    This version uses fast non-overlapping validation
    because STRIDE == PATCH_SIZE.
    """

    _, _, D, H, W = image.shape

    pd, ph, pw = patch_size

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

    model.eval()

    total = (
        len(d_positions)
        * len(h_positions)
        * len(w_positions)
    )

    with torch.no_grad():

        for d in d_positions:

            for h in h_positions:

                for w in w_positions:

                    patch = image[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw,
                    ].to(
                        device,
                        non_blocking=True,
                    )

                    # ----------------------------------------
                    # AMP inference
                    # ----------------------------------------

                    if device.type == "cuda":

                        with torch.amp.autocast(
                            device_type="cuda"
                        ):
                            logits = model(patch)

                    else:

                        logits = model(patch)

                    logits_sum[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw,
                    ] += logits.float()

                    count_map[
                        :,
                        :,
                        d:d + pd,
                        h:h + ph,
                        w:w + pw,
                    ] += 1.0

    logits_average = (
        logits_sum /
        count_map.clamp_min(1.0)
    )

    prediction = torch.argmax(
        logits_average,
        dim=1,
    )

    prediction = prediction.squeeze(0)

    return prediction.cpu().numpy()


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

        optimizer.zero_grad(
            set_to_none=True
        )

        # ====================================================
        # AMP TRAINING
        # ====================================================

        if scaler is not None:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                logits = model(images)

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

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

            loss.backward()

            optimizer.step()

        # ====================================================
        # TRAIN DICE
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
# FAST FULL-VOLUME VALIDATION
# ============================================================

def validate(
    model,
    dataloader,
    criterion,
    device,
    epoch,
):

    model.eval()

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

        for batch_index, batch in enumerate(pbar):

            images = batch["image"]

            masks = batch["mask"]

            # ------------------------------------------------
            # Validation batch size = 1
            # ------------------------------------------------

            if images.ndim == 4:
                images = images.unsqueeze(0)

            if masks.ndim == 3:
                masks = masks.unsqueeze(0)

            original_shape = images.shape[2:]

            # ------------------------------------------------
            # Pad if necessary
            # ------------------------------------------------

            images_padded, _ = pad_volume(
                images,
                PATCH_SIZE,
            )

            # ------------------------------------------------
            # FULL 3D PREDICTION
            # ------------------------------------------------

            prediction_np = sliding_window_prediction(
                model=model,
                image=images_padded,
                device=device,
                patch_size=PATCH_SIZE,
                stride=STRIDE,
            )

            # ------------------------------------------------
            # Remove padding
            # ------------------------------------------------

            D, H, W = original_shape

            prediction_np = prediction_np[
                :D,
                :H,
                :W,
            ]

            predictions = torch.from_numpy(
                prediction_np
            ).long().to(device)

            masks_device = masks.to(device)

            # =================================================
            # METRICS
            # =================================================

            mean_dice, _ = compute_dice_score(
                predictions.unsqueeze(0),
                masks_device,
            )

            mean_iou, _ = compute_iou_score(
                predictions.unsqueeze(0),
                masks_device,
            )

            brats_metrics = (
                compute_brats_regions_metrics(
                    predictions.unsqueeze(0),
                    masks_device,
                )
            )

            dice_tracker.update(
                mean_dice,
                1,
            )

            iou_tracker.update(
                mean_iou,
                1,
            )

            wt_tracker.update(
                brats_metrics["dice_wt"],
                1,
            )

            tc_tracker.update(
                brats_metrics["dice_tc"],
                1,
            )

            et_tracker.update(
                brats_metrics["dice_et"],
                1,
            )

            # =================================================
            # SAVE FIRST VALIDATION PATIENT
            # =================================================

            if batch_index == 0:

                sample_for_viz = (
                    images[0].cpu().numpy(),
                    masks[0].cpu().numpy(),
                    prediction_np,
                )

            pbar.set_postfix(
                {
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
        "val_loss": 0.0,

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
            "using REAL BraTS 2023 data"
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
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
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

    cfg.train.early_stopping_patience = args.patience

    # --------------------------------------------------------
    # Data directory
    # --------------------------------------------------------

    if args.data_dir is not None:

        cfg.dataset.data_dir = args.data_dir

    # ========================================================
    # CHECK DATASET
    # ========================================================

    if not os.path.exists(
        cfg.dataset.data_dir
    ):

        raise FileNotFoundError(
            "\nBraTS dataset directory was not found:\n"
            f"{cfg.dataset.data_dir}\n\n"
            "Please check config.py or use --data-dir."
        )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()

    print("=" * 70)

    print(
        "3D ATTENTION U-NET TRAINING"
    )

    print("=" * 70)

    print(
        "DATA TYPE       : REAL BraTS 2023"
    )

    print(
        "GROUND TRUTH    : *-seg.nii.gz"
    )

    print(
        "Device          :",
        device,
    )

    print(
        "Training patch  :",
        PATCH_SIZE,
    )

    print(
        "Validation stride:",
        STRIDE,
    )

    print(
        "Validation frequency:",
        f"Every {VALIDATE_EVERY} epochs",
    )

    print(
        "Classes         :",
        NUM_CLASSES,
    )

    print(
        "Train patients  :",
        MAX_TRAIN_PATIENTS,
    )

    print(
        "Val patients    :",
        MAX_VAL_PATIENTS,
    )

    print(
        "Data directory  :",
        cfg.dataset.data_dir,
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
        "Starting REAL BraTS 2023 training"
    )

    # ========================================================
    # DATASET SPLIT
    # ========================================================

    print()

    print(
        "Creating train/validation/test split..."
    )

    train_dirs, val_dirs, test_dirs = (
        create_train_val_test_split(
            data_dir=cfg.dataset.data_dir,
            val_split=cfg.dataset.val_split,
            test_split=cfg.dataset.test_split,
            seed=cfg.dataset.seed,
        )
    )

    # ========================================================
    # LIMIT DATASET
    # ========================================================

    train_dirs = train_dirs[
        :MAX_TRAIN_PATIENTS
    ]

    val_dirs = val_dirs[
        :MAX_VAL_PATIENTS
    ]

    # ========================================================
    # CHECK DATA
    # ========================================================

    if len(train_dirs) == 0:

        raise RuntimeError(
            "No real BraTS training patients found."
        )

    if len(val_dirs) == 0:

        raise RuntimeError(
            "No real BraTS validation patients found."
        )

    # ========================================================
    # PATIENT COUNTS
    # ========================================================

    print()

    print(
        "Training patients  :",
        len(train_dirs),
    )

    print(
        "Validation patients:",
        len(val_dirs),
    )

    print(
        "Test patients      :",
        len(test_dirs),
    )

    print()

    print(
        "Ground-truth masks:"
    )

    print(
        "Each patient uses its corresponding "
        "*-seg.nii.gz file."
    )

    # ========================================================
    # TRANSFORMS
    # ========================================================

    train_transform = ComposeTransforms(
        [

            ZScoreNormalize(),

            RandomCrop3D(
                patch_size=PATCH_SIZE
            ),

            RandomFlip3D(
                prob=0.5
            ),

            ToTensor(),

        ]
    )

    # --------------------------------------------------------
    # NO RANDOM CROP FOR VALIDATION
    # --------------------------------------------------------

    val_transform = ComposeTransforms(
        [

            ZScoreNormalize(),

            ToTensor(),

        ]
    )

    # ========================================================
    # DATASETS
    # ========================================================

    print()

    print(
        "Loading REAL BraTS training dataset..."
    )

    train_ds = BraTSDataset3D(
        train_dirs,
        modalities=cfg.dataset.modalities,
        transform=train_transform,
        is_train=True,
    )

    print(
        "Loading REAL BraTS validation dataset..."
    )

    val_ds = BraTSDataset3D(
        val_dirs,
        modalities=cfg.dataset.modalities,
        transform=val_transform,
        is_train=False,
    )

    print()

    print(
        "Train dataset size:",
        len(train_ds),
    )

    print(
        "Validation dataset size:",
        len(val_ds),
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
    # MODEL
    # ========================================================

    print()

    print(
        "Creating Attention U-Net 3D..."
    )

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
    # TRAINING LOOP
    # ========================================================

    best_val_dice = 0.0

    patience_counter = 0

    last_val_metrics = {
        "val_dice": 0.0,
        "val_iou": 0.0,
        "val_dice_wt": 0.0,
        "val_dice_tc": 0.0,
        "val_dice_et": 0.0,
        "sample_viz": None,
    }

    print()

    print("=" * 70)

    print(
        "STARTING TRAINING"
    )

    print("=" * 70)

    print()

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
        #
        # Only every N epochs.
        # Always validate on the final epoch.
        # ====================================================

        should_validate = (
            epoch % VALIDATE_EVERY == 0
            or epoch == cfg.train.epochs
        )

        if should_validate:

            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
            )

            last_val_metrics = val_metrics

        else:

            val_metrics = last_val_metrics

            print()

            print(
                f"Skipping validation for epoch {epoch}. "
                f"Next validation at epoch "
                f"{((epoch // VALIDATE_EVERY) + 1) * VALIDATE_EVERY}."
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
        # LOG RESULTS
        # ====================================================

        logger.info(
            f"Epoch [{epoch:03d}/"
            f"{cfg.train.epochs:03d}] "
            f"({elapsed:.1f}s) | "

            f"Train Loss: "
            f"{train_metrics['loss']:.4f} | "

            f"Train Dice: "
            f"{train_metrics['dice']:.4f} | "

            f"Val Dice: "
            f"{val_metrics['val_dice']:.4f}"
        )

        print()

        print(
            f"Epoch {epoch}"
        )

        print(
            f"Train Loss : "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Train Dice : "
            f"{train_metrics['dice']:.4f}"
        )

        if should_validate:

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

        if should_validate:

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
        #
        # Only update validation checkpoint when validation
        # actually happened.
        # ====================================================

        if should_validate:

            is_best = ckpt_manager.step(
                current_metric=val_metrics["val_dice"],

                epoch=epoch,

                model=model,

                optimizer=optimizer,

                scheduler=scheduler,
            )

            # =================================================
            # BEST MODEL
            # =================================================

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
                    "⭐ BEST MODEL SAVED"
                )

                print(
                    f"Best Validation Dice: "
                    f"{best_val_dice:.4f}"
                )

                # --------------------------------------------
                # Visualization
                # --------------------------------------------

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

            # -----------------------------------------------
            # NO IMPROVEMENT
            # -----------------------------------------------

            else:

                patience_counter += 1

                print()

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

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 70)

    print(
        f"Best Validation Dice: "
        f"{best_val_dice:.4f}"
    )

    print(
        "Training patients:",
        len(train_dirs),
    )

    print(
        "Validation patients:",
        len(val_dirs),
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