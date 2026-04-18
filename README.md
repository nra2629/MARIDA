# MARIDA: Marine Debris Detection from Sentinel-2 Satellite Imagery
### Deep Learning with PyTorch — Multi-Label Image Classification

---

## Project Overview

Marine plastic pollution is one of the most critical environmental challenges of our time.
This project builds an end-to-end deep learning pipeline to automatically detect **marine debris
and 14 other ocean surface features** from multispectral satellite images captured by the
European Space Agency's Sentinel-2 satellite.

**Unstructured image data** processed with **PyTorch deep learning** — no CSV or tabular data.

### What the model does
Given a 256×256 pixel satellite patch with 11 spectral bands, the CNN predicts which
of 15 ocean surface classes are present (multi-label classification):

| Class | Class |
|-------|-------|
| Marine Debris | Dense Sargassum |
| Sparse Sargassum | Natural Organic Material |
| Ship | Clouds |
| Marine Water | Sediment-Laden Water |
| Foam | Turbid Water |
| Shallow Water | Waves |
| Cloud Shadows | Wakes |
| Mixed Water | |

---

## Project Structure

```
marida_1/
├── MARIDA_Project.ipynb   ← Main notebook (fully self-contained)
├── README.md              ← This file
├── environment.yml        ← Conda environment (recommended)
├── requirements.txt       ← Pip requirements (alternative)
└── outputs/
    └── cnn/
        ├── best_model.pt  ← Saved CNN checkpoint
        └── history.json   ← Training history
```

> **Note on data:** The dataset files (TIF images) are large (~10GB) and are **not included**
> in this repository. See the Data Setup section below to download them.

---

## Environment Setup

### Option A — Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate marida_env
jupyter notebook
```

### Option B — Pip

```bash
pip install -r requirements.txt
jupyter notebook
```

### Verify PyTorch and device
```python
import torch
print(torch.__version__)
print(torch.backends.mps.is_available())  # True on Apple Silicon Mac
```

---

## Data Setup

The MARIDA dataset is publicly available here:
**https://zenodo.org/record/5823383**

After downloading, the data folder structure looks like this:
```
data/patches/
├── labels_mapping.txt          ← JSON: image filename → 15-class label
├── train_X.txt                 ← list of training tile IDs
├── val_X.txt
├── test_X.txt
└── S2_<scene>/
    ├── S2_<tile-id>.tif        ← 11-band GeoTIFF image (256×256)
    ├── S2_<tile-id>_cl.tif     ← classification mask
    └── S2_<tile-id>_conf.tif   ← confidence mask
```

> **Important:** The TIF files are organised in subfolders per scene
> (e.g. `S2_27-1-19_16PCC/S2_27-1-19_16PCC_29.tif`).
> The notebook handles this automatically using the find_tif() helper.

Once downloaded, update the DATA variable in Cell 0 of the notebook:
```python
DATA = "/path/to/your/data/patches"
```

---

## Running the Analysis

1. Clone this repository
2. Set up the environment (see above)
3. Download the data from Zenodo and update the DATA path in Cell 0
4. Launch Jupyter: `jupyter notebook`
5. Open `MARIDA_Project.ipynb`
6. Run cells top to bottom with Shift+Enter

---

## Notebook Contents

| Section | Description |
|---------|-------------|
| 0. Setup | Imports, device selection (MPS/CUDA/CPU) |
| 1. Dataset Description | MARIDA overview, 15 class names |
| 2. File Finder | Locates TIF files in subfolders |
| 3. Custom Dataset | PyTorch Dataset with normalisation |
| 4. Augmentation | Band-agnostic transforms |
| 5. Load Data | Train/val/test splits |
| 6. EDA | Class frequency, sample tiles, spectral profiles |
| 7. CNN Architecture | Residual CNN with skip connections |
| 8. Training Functions | Loss, optimiser, LR schedule, early stopping |
| 9. Train CNN | Live epoch table + learning curves |
| 10. Evaluation | Test set F1, mAP, per-class AP, PR curves |
| 11. GradCAM | Spatial attention heatmaps |
| 12. SHAP | Spectral band importance |
| 13. Discussion | Results, limitations, conclusions |

---

## Key Design Decisions

**Loss:** BCEWithLogitsLoss with pos_weight — rare classes penalised more when missed.

**Normalisation:** Per-channel z-score on training set only — no data leakage to val/test.

**Residual CNN:** Skip connections solve vanishing gradients. Downsamples 256×256 → 4×4 → pooled.

**Augmentation:** Random flips, rotations, noise, cutout — band-agnostic (works for 11 bands).

**MLP excluded:** Flattened input of 720,896 features exceeded Apple Silicon GPU memory.

---

## Hardware

| Device | CNN Training Time |
|--------|------------------|
| Apple MPS (M1/M2/M3) | ~45–60 minutes |
| NVIDIA GPU | ~20 minutes |
| CPU only | ~3 hours |

---

## Results

| Metric | CNN |
|--------|-----|
| Macro F1 | 0.1954 |
| Micro F1 | 0.2198 |
| mAP | 0.1239 |

Best class: Marine Water (AP = 0.630)
Hardest classes: Dense Sargassum / Waves (AP = 0.036)

Low scores on rare classes are expected — up to 19x imbalance between classes.

---

## References

- Kikaki et al. (2022). MARIDA: A benchmark for Marine Debris detection. PLOS ONE. https://doi.org/10.1371/journal.pone.0262247
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
