# Coronary Artery Disease Detection from Cardiac MRI

A deep learning system that classifies cardiac MRI scans as **Normal** or **CAD-positive** (Coronary Artery Disease) using transfer learning on EfficientNet-B0.

---

## Dataset

| Property | Details |
|---|---|
| Source | CAD Cardiac MRI Dataset (Danial Sharifrazi) |
| Patients | 30 total — 16 Normal, 14 Sick |
| Images | 63,151 unique JPEG slices |
| Structure | `Normal/Directory_1–16/` and `Sick/Directory_17–30/` |

Each `Directory_X` is one patient. Each patient contains multiple MRI series subdirectories, each holding individual cardiac slice images.

---

## Model Architecture

**Base**: EfficientNet-B0 (ImageNet pretrained, frozen during Phase 1)

**Custom classifier head** (replaces the original):
```
BatchNorm1d(1280) → Dropout(0.3) → Linear(1280, 256) → ReLU → Dropout(0.2) → Linear(256, 1) → Sigmoid
```

**Two-phase training**:
- **Phase 1** (15 epochs, lr=1e-3): Train classifier head only, backbone frozen
- **Phase 2** (10 epochs, lr=1e-5): Fine-tune top 3 EfficientNet blocks + head

**Loss**: Class-weighted BCE (handles Normal/Sick imbalance)  
**Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)  
**Early stopping**: patience=5 epochs on val loss

---

## Data Split

Patient-level stratified split — no patient appears in more than one split:

| Split | Normal patients | Sick patients | Total patients | Slices (~) |
|---|---|---|---|---|
| Train | 12 | 10 | 22 | 45,000 |
| Val | 2 | 2 | 4 | 9,000 |
| Test | 2 | 2 | 4 | 8,700 |

> **Why patient-level?** Slice-level random splits cause data leakage — slices from the same patient appear in both train and test, inflating metrics artificially.

---

## Results

> Run `python evaluate.py` after training to reproduce these results.

| Metric | Score |
|---|---|
| Accuracy | — |
| Sensitivity (Recall) | — |
| Specificity | — |
| Precision | — |
| F1 Score | — |
| ROC AUC | — |

*Evaluation plots saved to `static/`: `confusion_matrix.png`, `roc_pr_curves.png`, `training_history.png`*

---

## Project Structure

```
├── app.py                  # Flask web app — /predict, /gradcam, /health
├── model_arch.py           # EfficientNet-B0 architecture definition
├── preprocess.py           # Transforms, MRIDataset, patient splitting utilities
├── train.py                # Training loop, evaluation functions
├── evaluate.py             # Standalone evaluation script
├── kaggle_train.ipynb      # Kaggle notebook (imports local .py files via %%writefile)
├── requirements.txt
├── model/
│   └── cad_model.pth       # Trained model weights
├── static/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_pr_curves.png
├── templates/
│   └── index.html          # PACS-style dark UI
└── demo_data/              # Sample images for live demo
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset at `C:\Users\<you>\Downloads\dataset` with `Normal\` and `Sick\` subdirectories, or update `DATASET_PATH` in `train.py` and `evaluate.py`.

---

## Training

**Locally** (no GPU — slow):
```bash
python train.py
```

**On Kaggle** (recommended — free T4 GPU):
1. Open `kaggle_train.ipynb` on Kaggle
2. Add the CAD dataset: `Settings → Add Data`
3. Enable GPU: `Settings → Accelerator → GPU T4 x2`
4. Run all cells (~30–60 min)
5. Download `cad_model.pth` from the Output tab
6. Place it at `model/cad_model.pth`

---

## Evaluation

```bash
python evaluate.py
```

Outputs confusion matrix, ROC curve, and Precision-Recall curve to `static/`.

---

## Web App

```bash
python app.py
```

Open `http://localhost:5000` — upload one or more cardiac MRI slices for prediction. Features:
- Multi-image upload with drag & drop
- Per-slice CAD probability
- Study-level aggregation
- Grad-CAM heatmap overlay
- PACS-style dark UI

---

## Limitations

- Only 30 patients total — too few for clinical-grade generalisation claims
- No external validation cohort
- Results should be interpreted as a proof-of-concept, not a clinical tool
