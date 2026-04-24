# WikiArt Painting Classification — Deep Learning Project

**Group 22 · Deep Learning 2025/2026**

| Name | Student ID |
|---|---|
| Afonso Hermenegildo | 20221958 |
| André Ferreira | 20250398 |
| André Nicolau | 20221918 |
| Lara Santos | 20221823 |
| Marco Martins | 20221951 |

---

## Overview

Multi-class image classification of paintings by artist using a modified subset of the [WikiArt](https://www.wikiart.org/) dataset (23 artists/classes). Two model families were developed and compared: a **custom CNN built from scratch** and **fine-tuned pre-trained models** (VGG16, ResNet, EfficientNet, ConvNeXt).

---

## Repository Structure

```
Wikiart_Project/
├── Data/                          # Raw dataset (Train / Validation / Test splits)
├── clean_split_data/              # Stratified splits after data cleaning
│   ├── Train/
│   ├── Validation/
│   └── Test/
├── outputs/
│   ├── models/                    # Saved .keras model files
│   └── figures/                   # Generated plots and visualisations
│
├── notebooks/
│   ├── eda.ipynb                      # Exploratory Data Analysis
│   ├── split_data.ipynb               # Stratified train/val/test split
│   ├── base_model.ipynb               # Custom CNN experiments
│   ├── augmentation_exploration_executed.ipynb  # Augmentation pipeline comparison
│   ├── pretrained_models.ipynb        # Pre-trained model fine-tuning
│   ├── evaluation_baseline.ipynb      # Evaluation of the custom CNN
│   └── evaluation_pretrained.ipynb    # Evaluation of the best pre-trained model
│
├── scripts/
│   ├── augmentation.py                # Data augmentation pipelines
│   ├── base_model_class.py            # Reusable Keras Model subclass
│   ├── eda_utils.py                   # EDA helper functions
|   └── models_utils.py                # Training history plotting utilities
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/MarcoAFMartins/Wikiart_Project.git
cd Wikiart_Project
```

### 2. Install dependencies

Python 3.10+ is recommended. Install all required packages with:

```bash
pip install -r requirements.txt
```

Key dependencies: `tensorflow==2.21.0`, `keras==3.13.2`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`, `pillow`, `imagehash`.

### 3. Add the dataset

Place the WikiArt dataset inside the `Data/` directory, organised as one sub-folder per artist:

```
Data/
└── <artist_name>/
    ├── painting_001.jpg
    ├── painting_002.jpg
    └── ...
```

---

## Workflow

Run the notebooks **in order**:

| Step | Notebook | Description |
|---|---|---|
| 1 | `eda.ipynb` | Dataset statistics, class balance, colour analysis, quality checks (duplicates, corrupted files, grayscale images, brightness outliers) |
| 2 | `split_data.ipynb` | Stratified 70/15/15 train/validation/test split → `clean_split_data/` |
| 3 | `augmentation_exploration_executed.ipynb` | Visual comparison of the six augmentation pipelines |
| 4 | `base_model.ipynb` | Iterative custom CNN development (SGD → Adam, dropout, batch normalisation, global average pooling) |
| 5 | `pretrained_models.ipynb` | VGG16, ResNet, EfficientNet, ConvNeXt — feature extraction and fine-tuning |
| 6 | `evaluation_baseline.ipynb` | Classification report, confusion matrix, Grad-CAM, misclassified samples for the custom CNN |
| 7 | `evaluation_pretrained.ipynb` | Same evaluation suite for the best pre-trained model (ConvNeXt Large fine-tuned) |

---

## Models

### Custom CNN (`base_model.ipynb`)

Built and refined iteratively from a basic convolutional network to a deeper architecture featuring:
- 4 convolutional blocks (filters: 40 → 40 → 64 → 64)
- Batch Normalisation before each activation
- Global Average Pooling
- Dense head with Dropout (0.2)
- Input image size: **128 × 128**
- Optimizer: Adam · Loss: Categorical Cross-Entropy

### Pre-trained Models (`pretrained_models.ipynb`)

Four ImageNet-pre-trained backbones were evaluated in two modes:
- **Feature extraction** — backbone frozen, only the classification head trained
- **Fine-tuning** — partial or full backbone unfreezing

Backbones compared: VGG16, ResNet, EfficientNet, **ConvNeXt Large** (best performing).  
Best model: `convnext_large_finetuned.keras` — input size **512 × 512**.

---

## Data Augmentation

Six painting-aware augmentation pipelines are defined in `augmentation.py`, ranging from conservative to aggressive. All pipelines respect the constraint that paintings are not photographs — heavy geometric distortions (large rotations, shear) can destroy style-defining features:

| Pipeline | Transforms |
|---|---|
| `augmentation_conservative` | Horizontal flip, tiny brightness jitter |
| `augmentation_mild` | + small rotation (≤ 5°) |
| `augmentation_moderate` | + contrast variation, light zoom (±5%) |
| `augmentation_moderate_plus` | + small translation (±5%) |
| `augmentation_aggressive` | Stronger versions of all above |
| `augmentation_moderate_noise` | `moderate_plus` + Gaussian noise (simulates scan artefacts) |

---

## Evaluation

Both models are evaluated on the held-out test set using:
- **Classification report** — per-class precision, recall, F1-score and macro averages
- **Confusion matrix** — seaborn heatmap
- **Training curves** — loss and accuracy over epochs
- **Grad-CAM** — gradient-weighted class activation maps to inspect model attention
- **Misclassified sample analysis** — visual inspection of errors

---

## License

For academic use only as part of the Deep Learning course (2025/2026).
