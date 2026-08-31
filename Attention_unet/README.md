# Attention U-Net for Multimodal Brain Tumor Segmentation (BraTS 2023)

A complete, production-grade PyTorch implementation of **2D and 3D Attention U-Net** for multi-modal brain tumor segmentation using the **BraTS 2023 Dataset**.

---

## 📌 Project Overview

Brain tumor segmentation from multi-modal Magnetic Resonance Imaging (MRI) is essential for computer-aided diagnosis, surgical planning, and quantitative monitoring of neuro-oncology patients. This project implements a **Modular Attention U-Net** from scratch in PyTorch, leveraging **Additive Soft Attention Gates** to automatically suppress feature responses in irrelevant brain tissue regions while emphasizing tumor sub-regions.

### Key Features
- **Framework**: PyTorch 2.0+
- **Input MRI Modalities (4 Channels)**: T1 Native (`t1n`), T1 Post-Contrast / T1ce (`t1c`), T2-Weighted (`t2w`), and T2-FLAIR (`t2f`).
- **Segmentation Output Classes (4 Classes)**:
  1. `Background` (Class 0)
  2. `Necrotic & Non-Enhancing Tumor Core (NCR)` (Class 1)
  3. `Peritumoral Edema (ED)` (Class 2)
  4. `Enhancing Tumor (ET)` (Class 3)
- **BraTS Clinical Evaluation Sub-Regions**:
  - **Whole Tumor (WT)**: Labels 1 + 2 + 3
  - **Tumor Core (TC)**: Labels 1 + 3
  - **Enhancing Tumor (ET)**: Label 3
- **Dual Architecture Support**: Full 2D and 3D Volumetric Attention U-Net implementations.
- **Robust Medical Preprocessing**: Z-score non-zero intensity normalization, random cropping, spatial flipping/rotations, and intensity jittering.
- **Mixed Precision & Optimization**: Automatic Mixed Precision (AMP), AdamW optimizer, Cosine Annealing learning rate scheduler, Early Stopping, and Checkpoint Management.
- **Synthetic Data Demo Mode**: Built-in synthetic brain generator for instant local code verification without requiring 100GB+ datasets.

---

## 📁 Repository Structure

```
Attention U-Net/
├── dataset/
│   ├── __init__.py
│   ├── brats_dataset.py       # BraTS 2D/3D NIfTI/NPZ Dataset & Synthetic Generator
│   └── transforms.py          # Medical Z-Score Normalization & Spatial Augmentations
├── models/
│   ├── __init__.py
│   ├── building_blocks.py     # DoubleConv, 2D/3D Additive Attention Gates, UpBlocks
│   ├── attention_unet_2d.py   # 2D Attention U-Net Model
│   └── attention_unet_3d.py   # 3D Volumetric Attention U-Net Model
├── losses/
│   ├── __init__.py
│   └── losses.py              # Soft Dice Loss, BCE, Combined Dice+BCE, Focal Loss
├── metrics/
│   ├── __init__.py
│   └── metrics.py             # Dice, IoU, PR/F1, BraTS Sub-Regions (WT/TC/ET), HD95
├── utils/
│   ├── __init__.py
│   ├── logger.py              # Console Logger, MetricTracker, TensorBoard Logger
│   ├── visualization.py       # Multi-modal MRI plot grid & RGBA Tumor Overlays
│   └── checkpoint.py          # Model Checkpoint saving, loading & Best Model tracking
├── configs/
│   ├── __init__.py
│   └── config.py              # Dataclass Master Configuration
├── checkpoints/               # Directory storing best_model.pth & latest_checkpoint.pth
├── logs/                      # Training logs & TensorBoard event files
├── results/                   # Output visualization plots & test metrics report
├── train.py                   # Master Training and Validation Pipeline
├── test.py                    # Dataset Evaluation Script
├── predict.py                 # Single-Volume / Batch Inference Pipeline
├── requirements.txt           # Python Dependencies
└── README.md                  # Project Documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/your-username/Attention-UNet-BraTS2023.git
cd Attention-UNet-BraTS2023

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup (BraTS 2023)

Organize your BraTS 2023 dataset directory structure as follows:
```
data/BraTS2023/
├── BraTS-GLI-00001-000/
│   ├── BraTS-GLI-00001-000-t1n.nii.gz
│   ├── BraTS-GLI-00001-000-t1c.nii.gz
│   ├── BraTS-GLI-00001-000-t2w.nii.gz
│   ├── BraTS-GLI-00001-000-t2f.nii.gz
│   └── BraTS-GLI-00001-000-seg.nii.gz
├── BraTS-GLI-00002-000/
│   ...
```

*Note: If local dataset directory is not present, the pipeline automatically triggers synthetic demo dataset generation for testing.*

---

## 🧠 Model Architecture & Attention Mechanism

### Additive Soft Attention Gate (Oktay et al., 2018)
The Attention Gate filters skip connections $x_l$ using gating signal $g$ from deeper decoder layers:

$$\psi_l = \text{ReLU}\left(W_g^T g + W_x^T x_l + b_g + b_x\right)$$

$$\alpha_l = \text{Sigmoid}\left(W_\psi^T \psi_l + b_\psi\right)$$

$$\hat{x}_l = x_l \odot \alpha_l$$

Where $W_g, W_x, W_\psi$ are $1\times 1(\times 1)$ linear convolutions, and $\alpha_l$ is the scalar attention coefficient map.

---

## 🚀 Training Instructions

### Run 3D Volumetric Training
```bash
python train.py --dim 3d --epochs 100 --batch-size 2 --lr 1e-4 --data-dir ./data/BraTS2023
```

### Run 2D Training
```bash
python train.py --dim 2d --epochs 50 --batch-size 8 --lr 1e-4 --data-dir ./data/BraTS2023
```

### Fast Demo Run (Synthetic Data)
```bash
python train.py --dim 3d --epochs 5 --synthetic --num-synthetic 4
```

---

## 🧪 Testing & Evaluation

Run evaluation on the test set using the best saved checkpoint (`best_model.pth`):
```bash
python test.py --dim 3d --checkpoint ./checkpoints/best_model.pth --data-dir ./data/BraTS2023
```

---

## 🔮 Single-Volume Inference

Predict tumor segmentation masks on a single patient volume:
```bash
python predict.py --dim 3d --checkpoint ./checkpoints/best_model.pth --patient-dir ./data/BraTS2023/BraTS-GLI-00001-000 --output-dir ./results/predictions
```

---

## 📊 Expected Performance Metrics

| Evaluation Metric | Whole Tumor (WT) | Tumor Core (TC) | Enhancing Tumor (ET) | Overall Mean |
| :--- | :---: | :---: | :---: | :---: |
| **Dice Similarity Coefficient** | `0.90 - 0.92` | `0.85 - 0.88` | `0.80 - 0.84` | `0.86` |
| **IoU (Jaccard Index)** | `0.82 - 0.85` | `0.74 - 0.79` | `0.67 - 0.72` | `0.76` |
| **HD95 (mm)** | `< 5.0 mm` | `< 7.0 mm` | `< 4.5 mm` | `< 5.5 mm` |

---

## 📄 License & References
- **BraTS 2023 Challenge**: Bakas et al., "Advancing The Cancer Genome Atlas brain MRI collection with consensus segmentations and clinical evaluations", *Scientific Data*, 2017.
- **Attention U-Net**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", *Medical Image Computing and Computer Assisted Intervention (MICCAI / MIDL)*, 2018.
