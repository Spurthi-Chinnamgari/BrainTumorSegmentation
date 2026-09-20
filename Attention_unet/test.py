"""
Testing and Evaluation Pipeline for Attention U-Net Brain Tumor Segmentation.
Loads trained best checkpoint, evaluates on test dataset, computes metrics, and generates visual comparisons.
"""

from typing import Dict, Any, List
import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    class tqdm:
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, *args, **kwargs):
            pass

from configs import get_default_config
from dataset import (
    BraTSDataset2D,
    BraTSDataset3D,
    create_train_val_test_split,
    generate_synthetic_brats_data,
    ZScoreNormalize,
    RandomCrop2D,
    RandomCrop3D,
    ToTensor,
    ComposeTransforms,
)
from models import AttentionUNet2D, AttentionUNet3D
from metrics import (
    compute_dice_score,
    compute_iou_score,
    compute_precision_recall_f1,
    compute_brats_regions_metrics,
    compute_hausdorff_distance_95,
)
from utils import setup_logger, load_checkpoint, save_prediction_comparison


def evaluate_test_set(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    results_dir: str,
) -> Dict[str, Any]:
    """
    Evaluate trained model on test dataset and save visual comparisons.
    """
    model.eval()

    all_dices = []
    all_ious = []
    all_wt_dices = []
    all_tc_dices = []
    all_et_dices = []
    all_hd95s = []

    os.makedirs(results_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating Test Set")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            mean_dice, class_dice = compute_dice_score(preds, masks)
            mean_iou, class_iou = compute_iou_score(preds, masks)
            brats_metrics = compute_brats_regions_metrics(preds, masks)

            all_dices.append(mean_dice)
            all_ious.append(mean_iou)
            all_wt_dices.append(brats_metrics["dice_wt"])
            all_tc_dices.append(brats_metrics["dice_tc"])
            all_et_dices.append(brats_metrics["dice_et"])

            # Compute HD95 for first sample in batch
            p_np = (preds[0] == 3).cpu().numpy()
            t_np = (masks[0] == 3).cpu().numpy()
            hd95 = compute_hausdorff_distance_95(p_np, t_np)
            all_hd95s.append(hd95)

            # Save first 5 test sample visual predictions
            if i < 5:
                img_np = images[0].cpu().numpy()
                tgt_np = masks[0].cpu().numpy()
                prd_np = preds[0].cpu().numpy()
                patient_id = batch.get("patient_id", [f"test_sample_{i}"])[0]
                save_path = os.path.join(results_dir, f"test_pred_{patient_id}.png")
                save_prediction_comparison(
                    img_np, tgt_np, prd_np, save_path=save_path, title=f"Test Sample: {patient_id}"
                )

    metrics_summary = {
        "test_mean_dice": float(np.mean(all_dices)),
        "test_mean_iou": float(np.mean(all_ious)),
        "test_dice_wt": float(np.mean(all_wt_dices)),
        "test_dice_tc": float(np.mean(all_tc_dices)),
        "test_dice_et": float(np.mean(all_et_dices)),
        "test_mean_hd95_et": float(np.mean(all_hd95s)),
    }

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Attention U-Net on BraTS 2023 Test Set")
    parser.add_argument("--dim", type=str, default="3d", choices=["2d", "3d"], help="Model dimension")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth", help="Checkpoint file path")
    parser.add_argument("--data-dir", type=str, default="./data/BraTS2023", help="Dataset directory")
    parser.add_argument("--synthetic", action="store_true", help="Evaluate on synthetic data")
    args = parser.parse_args()

    cfg = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(log_file=os.path.join(cfg.train.log_dir, "test.log"))

    logger.info("=" * 60)
    logger.info(f"Starting Attention U-Net ({args.dim.upper()}) Evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("=" * 60)

    # 1. Dataset Preparation
    if args.synthetic or not os.path.exists(args.data_dir):
        logger.info("[!] Using synthetic dataset for test evaluation...")
        synthetic_dir = "./data/BraTS2023_Synthetic"
        patient_dirs = generate_synthetic_brats_data(output_dir=synthetic_dir, num_patients=2, volume_shape=(32, 64, 64))
        test_dirs = patient_dirs
    else:
        _, _, test_dirs = create_train_val_test_split(
            data_dir=args.data_dir, val_split=cfg.dataset.val_split, test_split=cfg.dataset.test_split
        )

    logger.info(f"Test Patients Count: {len(test_dirs)}")

    # 2. Transformations & DataLoader
    if args.dim == "3d":
        transform = ComposeTransforms([ZScoreNormalize(), RandomCrop3D(patch_size=(16, 32, 32)), ToTensor()])
        test_ds = BraTSDataset3D(test_dirs, transform=transform)
    else:
        transform = ComposeTransforms([ZScoreNormalize(), RandomCrop2D(crop_size=(32, 32)), ToTensor()])
        test_ds = BraTSDataset2D(test_dirs, transform=transform, slice_dim=2, min_nonzero_ratio=0.0)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 3. Model & Checkpoint Loading
    if args.dim == "3d":
        model = AttentionUNet3D(in_channels=4, out_channels=4, features=[16, 32, 64, 128, 256]).to(device)
    else:
        model = AttentionUNet2D(in_channels=4, out_channels=4, features=[16, 32, 64, 128, 256]).to(device)

    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model=model, device=device)
        logger.info("Successfully loaded checkpoint weights.")
    else:
        logger.warning(f"Checkpoint {args.checkpoint} not found! Evaluating randomly initialized model.")

    # 4. Evaluation
    results = evaluate_test_set(model, test_loader, device, results_dir=cfg.train.results_dir)

    logger.info("-" * 60)
    logger.info("TEST SET EVALUATION RESULTS SUMMARY:")
    logger.info(f" Mean Dice Score:          {results['test_mean_dice']:.4f}")
    logger.info(f" Mean IoU Score:           {results['test_mean_iou']:.4f}")
    logger.info(f" Whole Tumor (WT) Dice:    {results['test_dice_wt']:.4f}")
    logger.info(f" Tumor Core (TC) Dice:     {results['test_dice_tc']:.4f}")
    logger.info(f" Enhancing Tumor (ET) Dice:{results['test_dice_et']:.4f}")
    logger.info(f" Mean HD95 (ET):           {results['test_mean_hd95_et']:.2f} mm")
    logger.info("-" * 60)

    # Save results summary JSON
    results_json_path = os.path.join(cfg.train.results_dir, "test_results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved test metrics report to: {results_json_path}")


if __name__ == "__main__":
    main()
