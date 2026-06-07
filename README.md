# ASL Fingerspelling Classification

Comparison of three models for American Sign Language (ASL) fingerspelling recognition: a classical Gabor+SVM baseline, a custom CNN trained from scratch, and a MobileNetV2 fine-tuned with progressive unfreezing.

**Dataset:** [ASL Fingerspelling Images (RGB & Depth)](https://www.kaggle.com/datasets/kuzivakwashe/significant-asl-alphabet-dataset) — 5 subjects, 24 static letters (A–Y, excluding J and Z).

| Model | Test Acc. | Val Acc. |
|---|---|---|
| Gabor + SVM | 22.8% | — |
| Custom CNN | 54.9% | 99.5% |
| MobileNetV2 | 81.5% | 99.9% |

*Test set = subject 5 (held out entirely). Val set = 10% of subjects 1–4.*

---

## Project structure

```
ASL-fingerspelling/
├── main.py                  # entry point: --model {gabor_svm|cnn_scratch|mobilenetv2}
├── comparison.py            # generates figures and comparison table from runs/
├── params.py                # loads doc/default.json and merges CLI overrides
├── doc/
│   ├── default.json         # all hyperparameters
│   └── requirements.txt
├── dataset/
│   ├── dataset.py           # ASLDataset (PyTorch Dataset)
│   ├── loaders.py           # get_loaders() — train/val/test DataLoaders
│   └── preprocessing/
│       ├── filters.py       # Gabor bank + feature extraction
│       └── image_processing.py
├── net/
│   ├── backbone/mobilenet.py
│   └── networks/
│       ├── cnn_scratch.py
│       └── mobilenetv2.py
├── phases/
│   ├── train.py             # train_one_ep()
│   └── infer.py             # infer_one_ep()
├── metrics/                 # accuracy, confusion matrix, per-class metrics
├── visual/plot.py           # confusion matrix plot
├── utility/                 # save/load checkpoint, get_fresh_model
├── figures/                 # output figures (committed)
└── runs/                    # training outputs (gitignored except results.json)
```

---

## Environment

Python 3.10+ is recommended. Two setup paths are supported:

### Local (CPU or NVIDIA GPU)

```bash
# 1. Clone the repo
git clone https://github.com/yeraay21/ASL-fingerspelling.git
cd ASL-fingerspelling

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install PyTorch (CUDA 12.1 — skip --index-url for CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install the rest of the dependencies
pip install -r doc/requirements.txt
```

### Google Colab

```python
# Cell 1 — install (run once per session)
!pip install scikit-image tqdm
!pip uninstall torch torchvision -y
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# Cell 2 — mount Drive and clone repo
from google.colab import drive
drive.mount("/content/drive")
%cd /content
!git clone https://github.com/yeraay21/ASL-fingerspelling.git
%cd ASL-fingerspelling

# Cell 3 — link dataset from Drive to expected path
import os
os.makedirs("data/raw", exist_ok=True)
!ln -sf /content/drive/MyDrive/DeepLearning/ASL_Project/data/raw/fingerspelling-asl \
         data/raw/fingerspelling-asl

```

---

## Dataset setup

Download the dataset from Kaggle and place it so the directory tree matches:

```
data/raw/fingerspelling-asl/
├── subject-1/
│   ├── a/   (RGB images)
│   ├── b/
│   ...
├── subject-2/
├── subject-3/
├── subject-4/
├── subject-5/
└── subject-1-extra/   (optional, included by default)
```

## Reproducing results

All hyperparameters live in [doc/default.json](doc/default.json). The key ones are:

| Parameter | Value |
|---|---|
| `img_size_small` | 64 × 64 (Gabor+SVM, CNN) |
| `img_size_mobile` | 224 × 224 (MobileNetV2) |
| `batch_size` | 64 (CNN), 32 in practice (MobileNetV2) |
| `epochs` | 25 (CNN scratch) |
| `seed` | 42 |

### 1 — Gabor + SVM

```bash
python main.py --model gabor_svm
```

- Builds a bank of 40 Gabor filters (5 frequencies × 8 orientations).
- Extracts mean + std per filter → 80-dimensional feature vector per image.

### 2 — CNN from scratch

```bash
python main.py --model cnn_scratch
```

### 3 — MobileNetV2 (progressive unfreezing)

```bash
python main.py --model mobilenetv2
```

Pretrained on ImageNet. Input normalized with ImageNet mean/std. Three training phases:

| Phase | Unfrozen layers | Learning rate | Epochs |
|---|---|---|---|
| head | classifier only | 1e-3 | 3 |
| partial | features[14:] + classifier | 1e-4 | 5 |
| full | all | 1e-5 | 5 |

A checkpoint is saved at the end of each phase under `runs/mobilenetv2_<timestamp>/models/`.

- **Requires a GPU** (recommended at batch_size=32).

### Generate figures and comparison table

After training all three models:

```bash
python comparison.py
```

Reads the most recent `runs/{model}_*/scores/results.json` for each model and writes to `figures/`:

- `confusion_<model>.png` — confusion matrix
- `history_mobilenetv2.png` — loss/accuracy curves by phase
- `gabor_per_class_f1.png` — per-class F1 bar chart
- `comparison_bar.png` — grouped bar chart of all models

---