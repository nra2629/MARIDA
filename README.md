# MARIDA: Marine Debris Detection from Sentinel-2 Satellite Imagery
### Deep Learning with PyTorch — Multi-Label Image Classification

## Project Overview

Marine plastic pollution is one of the most pressing environmental challenges today.
This project builds a deep learning pipeline to automatically detect marine debris
and other ocean surface features from multispectral satellite images captured by
the European Space Agency's Sentinel-2 satellite.

Given a 256×256 pixel satellite image patch with 11 spectral bands, the model predicts
which of 15 ocean surface classes are present. This is a multi-label classification
problem — a single patch can belong to multiple classes at the same time.

The 15 classes are: Marine Debris, Dense Sargassum, Sparse Sargassum, Natural Organic
Material, Ship, Clouds, Marine Water, Sediment-Laden Water, Foam, Turbid Water,
Shallow Water, Waves, Cloud Shadows, Wakes, and Mixed Water.

## Project Structure

```
marida_1/
├── MARIDA_Project.ipynb   
├── README.md              
├── environment.yml        
├── requirements.txt       
└── outputs/
    └── cnn/
        ├── best_model.pt  
        └── history.json   
```

## Data

The MARIDA dataset is publicly available at:
https://zenodo.org/record/5823383

After downloading, the folder will contain subfolders per scene like:
```
data/patches/
├── labels_mapping.txt
├── train_X.txt
├── val_X.txt
├── test_X.txt
└── S2_<scene>/
    ├── S2_<tile-id>.tif
    ├── S2_<tile-id>_cl.tif
    └── S2_<tile-id>_conf.tif
```

Once downloaded, update the DATA variable in Cell 0 of the notebook to point
to your local data folder.

## Environment Setup

Option A — Conda:
```bash
conda env create -f environment.yml
conda activate marida_env
jupyter notebook
```

Option B — Pip:
```bash
pip install -r requirements.txt
jupyter notebook
```

## Running the Notebook

1. Clone this repository
2. Download the data from the Zenodo link above
3. Update the DATA path in Cell 0
4. Run cells top to bottom with Shift+Enter

## Notebook Contents

| Section | Description |
|---------|-------------|
| 0. Setup | Imports and device selection |
| 1. Dataset Description | Overview and class names |
| 2. File Finder | Locates TIF files in subfolders |
| 3. Custom Dataset | PyTorch Dataset class |
| 4. Augmentation | Training transforms |
| 5. Load Data | Train, val, test splits |
| 6. EDA | Class distribution and sample visualisation |
| 7. CNN Architecture | Residual CNN with skip connections |
| 8. Training Functions | Loss, optimiser, scheduler |
| 9. Train CNN | Training with live progress |
| 10. Evaluation | F1, mAP, per-class AP, PR curves |
| 11. GradCAM | Where the model looks |
| 12. SHAP | Which spectral bands matter |
| 13. Discussion | Results and conclusions |

## Results

| Metric | Value |
|--------|-------|
| Macro F1 | 0.1954 |
| Micro F1 | 0.2198 |
| mAP | 0.1239 |

Best class: Marine Water (AP = 0.630)
Most challenging: Dense Sargassum and Waves (AP = 0.036)

## References

Kikaki et al. (2022). MARIDA: A benchmark for Marine Debris detection. PLOS ONE. https://doi.org/10.1371/journal.pone.0262247

Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.

Lundberg and Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.

He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
