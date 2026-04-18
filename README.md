# MARIDA: Marine Debris Detection from Sentinel-2 Satellite Imagery
### Deep Learning with PyTorch — Multi-Label Image Classification

---

## Project Overview

Marine plastic pollution is one of the most critical environmental challenges of our time.
This project builds an end-to-end deep learning pipeline to automatically detect **marine debris
and 14 other ocean surface features** from multispectral satellite images captured by the
European Space Agency's Sentinel-2 satellite.

**This is an unstructured image data project** using PyTorch deep learning on real satellite imagery.

### What the model does
Given a 256×256 pixel satellite patch with 11 spectral bands, the model predicts which
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
MARIDA/
├── MARIDA_Project.ipynb   ← Main notebook (fully self-contained)
├── README.md              ← This file
├── environment.yml        ← Conda environment (recommended)
├── requirements.txt       ← Pip requirements (alternative)
└── data/
    └── patches/
        ├── labels_mapping.txt
        ├── train_X.txt
        ├── val_X.txt
        ├── test_X.txt
        ├── S2_<tile-id>.tif         (11-band image)
        ├── S2_<tile-id>_cl.tif      (classification mask)
        └── S2_<tile-id>_conf.tif    (confidence mask)
```

**The notebook is fully self-contained** — all code is inside `MARIDA_Project.ipynb`.
No additional `.py` files are required.

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
python -m venv marida_env
source marida_env/bin/activate        # Mac/Linux
# marida_env\Scripts\activate         # Windows

pip install -r requirements.txt
jupyter notebook
```

### Verify setup
Open a Python terminal and run:
```python
import torch
print(torch.__version__)
print(torch.backends.mps.is_available())  # Should be True on Apple Silicon
```

---

## Running the Analysis

1. Clone or download this repository
2. Set up the environment (see above)
3. Place your data files in `data/patches/` (see Data Setup below)
4. Launch Jupyter: `jupyter notebook`
5. Open `MARIDA_Project.ipynb`
6. Run cells **top to bottom** with Shift+Enter

Each cell prints its own output immediately. You do not need to run the entire
notebook at once — you can stop and resume at any cell.

---

## Data Setup

The MARIDA dataset is publicly available at:
**https://zenodo.org/record/5823383**

After downloading, place files in `data/patches/`:
```
data/patches/
├── labels_mapping.txt     (JSON: filename → 15-class multi-hot label)
├── train_X.txt            (list of training tile IDs)
├── val_X.txt
├── test_X.txt
└── S2_*.tif               (GeoTIFF image patches)
```

---

## Notebook Contents

| Section | Description |
|---------|-------------|
| 0. Setup | Imports, device selection (MPS/CUDA/CPU) |
| 1. Dataset Description | MARIDA overview, class list |
| 2. Custom Dataset | PyTorch Dataset class with normalisation |
| 3. Augmentation | Band-agnostic transforms |
| 4. Load Data | Train/val/test split loading |
| 5. EDA | Class frequency, sample tiles, spectral profiles |
| 6. Model Architectures | MLP baseline + residual CNN |
| 7. Training Setup | Loss, optimiser, LR schedule, early stopping |
| 8. Train MLP | Baseline training + learning curves |
| 9. Train CNN | Main model training + learning curves |
| 10. Evaluation | Test set metrics: F1, mAP, per-class AP |
| 11. MLP vs CNN | Head-to-head comparison |
| 12. GradCAM | Spatial attention heatmaps |
| 13. SHAP | Spectral band importance |
| 14. Discussion | Results, limitations, conclusions |

---

## Key Design Decisions

### Multi-label loss with class balancing
`BCEWithLogitsLoss` with per-class `pos_weight` to penalise the model more for
missing rare classes like Marine Debris.

### Confidence masking
Pixels with `conf == 0` are zeroed out — only reliably annotated regions contribute to training.

### Normalisation
Per-channel z-score normalisation computed on training set only (prevents data leakage).

### LR schedule
Linear warmup (5 epochs) → cosine annealing. Prevents early instability and smoothly
decays learning rate.

### Residual connections
Skip connections in the CNN allow gradients to flow through deep layers, solving
the vanishing gradient problem.

---

## Hardware Notes

| Device | MLP Training | CNN Training |
|--------|-------------|-------------|
| Apple MPS GPU | ~10 mins | ~45 mins |
| NVIDIA GPU | ~5 mins | ~20 mins |
| CPU only | ~30 mins | ~3 hours |

To check your device, run Cell 0 in the notebook.

---

## Results (Example)

| Model | Macro F1 | Micro F1 | mAP |
|-------|----------|----------|-----|
| MLP Baseline | ~0.35 | ~0.50 | ~0.40 |
| CNN | ~0.50 | ~0.65 | ~0.55 |

*Actual results vary depending on number of training epochs and hardware.*

---

## References

- Kikaki, K. et al. (2022). *MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.* PLOS ONE. https://doi.org/10.1371/journal.pone.0262247
- Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Lundberg, S.M. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
