> Spanish version: [WhatDoNOW.md](WhatDoNOW.md)

# WhatDoNOW — per-member operational guide

Actionable roadmap for Toni, Corentin and Yeray. The canonical plan lives in [plan.md](plan.md); this document only answers "what do I have to write right now?".

Every task follows the same pattern: **What → Why → Goal → How → How to verify**.

---

## Current repo state

- PyTorch infrastructure built and validated against real data: `python main.py --model dummy` boots up, discovers **24 classes** (J and Z excluded because they require motion), and builds the loaders (train: 69,794, val: 7,755, test: 24,000 images in `subject-5`).
- Already done (do not touch): [`dataset/`](dataset/), [`utility/`](utility/), [`params.py`](params.py), [`main.py`](main.py), [`doc/`](doc/).
- Still pending: the `NotImplementedError` stubs distributed across [`net/`](net/), [`phases/`](phases/), [`metrics/`](metrics/), [`visual/`](visual/) and [`dataset/preprocessing/filters.py`](dataset/preprocessing/filters.py).

## Assignment (updated)

| Member | Track | Blocking for |
|---|---|---|
| **Toni** | CNN scratch + phases + metrics + error analysis | Yeray (MobileNet) — **critical path** |
| **Corentin** | Gabor + SVM + visual/plot + slides | Nobody during week 2 |
| **Yeray** | MobileNetV2 + comparison | — |

---

## Shared conventions (required reading)

1. **Always work on `main`.** `git pull` before starting; small, descriptive commits.
2. **Do not touch** [`dataset/`](dataset/), [`utility/`](utility/), [`params.py`](params.py), [`main.py`](main.py) beyond the functions explicitly assigned to you. If you need infrastructure changes, ping the group first.
3. **Images arrive as PyTorch tensors** from the DataLoaders: `(B, 3, 64, 64)` for CNN/Gabor or `(B, 3, 224, 224)` for MobileNet. Normalization is done. Nobody re-implements I/O.
4. **The test set (`subject-5`) is untouched until the end.** Only `infer_one_ep(..., validation=False)` evaluates it, once, after training.
5. **Same output schema for every model.** Save to `runs/{model}_{timestamp}/scores/results.json` with: `{model, num_classes, classes, test: {loss, acc, y_true, y_pred}, history?, best_val_acc?}`.
6. **24 classes, not 26.** `num_classes` is passed in at runtime via `len(classes)`. Do not hardcode.
7. **Lazy imports.** The [`utility/get_fresh_model.py`](utility/get_fresh_model.py) factory imports each model inside its own `if` branch, so a missing stub from another teammate never blocks you.

---

## Toni — CNN scratch + phases + metrics + error analysis

Recommended order (the first three are critical path for Yeray):

### Task T1 — `phases/train.py` → `train_one_ep`

- **What.** Fill in [`phases/train.py`](phases/train.py).
- **Why.** Called by `main.run_cnn_scratch` (lines 122-128 of [main.py](main.py)) and will also be called by Yeray's `run_mobilenetv2`. Without this, nobody trains.
- **Goal.** Function `train_one_ep(model, loader, optimizer, loss_fn, device) -> {"loss": float, "acc": float}`, averaged over the epoch.
- **How.** Pseudocode is already in the file's docstring. Outline:

```python
import torch

def train_one_ep(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return {"loss": total_loss / total, "acc": total_correct / total}
```

- **How to verify.** Once you also have T2 and T3, run `python main.py --model cnn_scratch --epochs 1 --batch_size 64`. It should print `[ep 01] train loss=… acc=…` with no exceptions.

### Task T2 — `phases/infer.py` → `infer_one_ep`

- **What.** Fill in [`phases/infer.py`](phases/infer.py).
- **Why.** Same as T1 but for val/test. Returns `y_true/y_pred` so downstream metrics and confusion matrix can consume them.
- **Goal.** `infer_one_ep(model, loader, loss_fn, device, validation=True) -> {"loss", "acc", "y_true", "y_pred"}` — the last two are lists of `int`.
- **How.**

```python
import torch

def infer_one_ep(model, loader, loss_fn, device, validation=True):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            preds = logits.argmax(1)
            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y).sum().item()
            total += x.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return {"loss": total_loss / total, "acc": total_correct / total,
            "y_true": y_true, "y_pred": y_pred}
```

- **How to verify.** Same as T1.

### Task T3 — `net/networks/cnn_scratch.py` → `CNNScratch`

- **What.** Fill in [`net/networks/cnn_scratch.py`](net/networks/cnn_scratch.py).
- **Why.** One of the three models we compare.
- **Goal.** A `nn.Module` that takes a `(B, 3, 64, 64)` tensor and returns logits `(B, num_classes)`. **No softmax** (CrossEntropyLoss applies it internally).
- **How.** Architecture is already fixed in the plan. PyTorch skeleton:

```python
import torch.nn as nn

class CNNScratch(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

- **How to verify.** After T1+T2+T3: `python main.py --model cnn_scratch --epochs 2 --batch_size 64`. Validation accuracy should rise (even slightly) between epoch 1 and 2 on CPU.

### Task T4 — `metrics/`

- **What.** Fill in [`metrics/accuracy.py`](metrics/accuracy.py), [`metrics/per_class.py`](metrics/per_class.py), [`metrics/confusion_matrix.py`](metrics/confusion_matrix.py).
- **Why.** Yeray needs them for `comparison.py` in week 3.
- **Goal.** Thin wrappers around `sklearn.metrics`. No clever logic.
- **How.**

```python
# accuracy.py
import numpy as np
def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())

# per_class.py
from sklearn.metrics import precision_recall_fscore_support
def per_class_metrics(y_true, y_pred, num_classes):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    return {"precision": p.tolist(), "recall": r.tolist(),
            "f1": f.tolist(), "support": s.tolist()}

# confusion_matrix.py
from sklearn.metrics import confusion_matrix as sk_cm
def confusion_matrix(y_true, y_pred, num_classes):
    return sk_cm(y_true, y_pred, labels=list(range(num_classes)))
```

- **How to verify.** Quick tests on dummy lists in a Python REPL. `accuracy([0,1,2], [0,1,1])` → `0.6666...`.

### Task T5 — Train the CNN end-to-end

- **What.** Run the full training in Colab (T4 GPU) and commit `runs/cnn_scratch_*/scores/results.json` back to the repo (pull before, push after).
- **Why.** It is one of the deliverables (minimum criterion: test acc ≥ 50%).
- **How.** In Colab after `pip install -r doc/requirements.txt`:

```bash
python main.py --model cnn_scratch --epochs 25 --batch_size 64 --lr 1e-3
```

- **How to verify.** `runs/cnn_scratch_*/scores/results.json` exists with `test.acc ≥ 0.50` (ideal). If you plateau at 30-40%, double-check augmentation and early-stop patience.

### Task T6 — Error analysis

- **What.** Identify the 3-5 most frequent confusions (typically M/N, S/A, U/V/R).
- **Why.** Part of the final comparative analysis.
- **How.** Once Yeray has `comparison.py` (week 3), add a section there. Otherwise drop a short paragraph into `presentation/error_analysis.md` with the top confused pairs taken from the CNN's `metrics.confusion_matrix(cm)`.

---

## Corentin — Gabor + SVM + visual/plot + slides

Independent track: nobody is waiting on you during week 2.

### Task C1 — `dataset/preprocessing/filters.py`

- **What.** Fill in [`dataset/preprocessing/filters.py`](dataset/preprocessing/filters.py).
- **Why.** Feature extractor for the classical baseline.
- **Goal.** Two pure functions:
  - `build_gabor_bank(frequencies, orientations) -> list[np.ndarray]`: returns `len(frequencies) * orientations` real 2D kernels (40 total with the default config: 5 frequencies × 8 orientations).
  - `extract_features(images_gray, filters) -> np.ndarray`: matrix `(N, F)` with `F = 2 * len(filters)` (mean + std of each filter response).
- **How.** Suggestion (not mandatory) using `skimage` and `scipy`:

```python
import numpy as np
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve

def build_gabor_bank(frequencies, orientations):
    bank = []
    for theta in np.linspace(0, np.pi, orientations, endpoint=False):
        for freq in frequencies:
            kernel = np.real(gabor_kernel(frequency=freq, theta=theta))
            bank.append(kernel)
    return bank

def extract_features(images_gray, filters):
    N = len(images_gray)
    F = 2 * len(filters)
    out = np.zeros((N, F), dtype=np.float32)
    for i, img in enumerate(images_gray):
        for j, k in enumerate(filters):
            resp = convolve(img, k, mode="reflect")
            out[i, 2*j]     = resp.mean()
            out[i, 2*j + 1] = resp.std()
    return out
```

- **How to verify.** In a quick REPL: load 5 64×64 images, build the bank, extract features. Final shape must be `(5, 80)` with no NaNs.

### Task C2 — Gabor+SVM logic in `main.run_gabor_svm`

- **What.** Fill in the body of `run_gabor_svm` in [main.py](main.py) (lines 86-93). It currently raises `NotImplementedError`.
- **Why.** Entry point for the Gabor+SVM model.
- **Goal.** Full pipeline: extract Gabor features → reduce dim → train SVM → evaluate on test → save `scores/results.json`.
- **How.** Concrete outline (no DataLoader: sklearn is not batched, iterate the dataset straight into NumPy):

```python
def run_gabor_svm(cfg, run_dir: Path) -> dict:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    from dataset.dataset import ASLDataset, discover_classes
    from dataset.preprocessing.image_processing import (
        read_image_bgr, to_grayscale, resize, to_float01,
    )
    from dataset.preprocessing.filters import build_gabor_bank, extract_features

    classes = discover_classes(cfg.data_dir, cfg.subjects_train[0])

    def load_gray(subjects, include_extra):
        ds = ASLDataset(cfg.data_dir, subjects, classes,
                        transform=lambda img: img,
                        include_extra=include_extra)
        X = np.zeros((len(ds), cfg.img_size_small, cfg.img_size_small), dtype=np.float32)
        y = np.zeros(len(ds), dtype=np.int64)
        for i, (img_rgb, label) in enumerate(ds):
            gray = to_grayscale(img_rgb)
            gray = resize(gray, cfg.img_size_small)
            X[i] = to_float01(gray)
            y[i] = label
        return X, y

    X_tr, y_tr = load_gray(cfg.subjects_train, cfg.include_extra)
    X_te, y_te = load_gray(cfg.subjects_test, include_extra=False)

    bank = build_gabor_bank(cfg.gabor.frequencies, cfg.gabor.orientations)
    F_tr = extract_features(X_tr, bank)
    F_te = extract_features(X_te, bank)

    scaler = StandardScaler().fit(F_tr)
    F_tr_s, F_te_s = scaler.transform(F_tr), scaler.transform(F_te)

    pca = PCA(n_components=cfg.gabor.pca_components, random_state=cfg.seed).fit(F_tr_s)
    Z_tr, Z_te = pca.transform(F_tr_s), pca.transform(F_te_s)

    grid = GridSearchCV(
        SVC(kernel=cfg.gabor.svm_kernel),
        param_grid={"C": cfg.gabor.svm_C_grid},
        cv=3, n_jobs=-1,
    ).fit(Z_tr, y_tr)

    y_pred = grid.best_estimator_.predict(Z_te)
    acc = float(accuracy_score(y_te, y_pred))
    p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)

    return {"model": "gabor_svm",
            "num_classes": len(classes), "classes": classes,
            "best_C": grid.best_params_["C"], "cv_best_score": float(grid.best_score_),
            "test": {"acc": acc, "precision_macro": float(p),
                     "recall_macro": float(r), "f1_macro": float(f),
                     "y_true": y_te.tolist(), "y_pred": y_pred.tolist()}}
```

- **How to verify.** `python main.py --model gabor_svm` produces `runs/gabor_svm_*/scores/results.json` with `test.acc` (reasonable target: 30-60%). The GridSearch typically takes several minutes.

> Note: if you haven't used sklearn before, ask Yeray for a 30-min walkthrough of `PCA + GridSearchCV + SVC`. It saves hours of debugging.

### Task C3 — `visual/plot.py`

- **What.** Fill in [`visual/plot.py`](visual/plot.py): `plot_history` and `plot_confusion_matrix`.
- **Why.** `comparison.py` (Yeray) and the slides consume them to generate the PNGs in `figures/`.
- **Goal.** Two functions that save a PNG to disk. Return nothing.
- **How.**

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_history(history, out_path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"]   for h in history], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"]   for h in history], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)

def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
```

- **How to verify.** Calling with dummy data produces a legible PNG.

### Task C4 — Slides skeleton

- **What.** Create `presentation/slides.pdf` (or `.pptx`).
- **Why.** Final deliverable alongside the code.
- **How.** Suggested structure (~12 slides):
  1. Cover (project, authors, course).
  2. Problem and dataset (ASL, 24 letters, 5 subjects, subject split).
  3. Why subject split matters (hand variability).
  4. Shared pipeline (preprocessing, augmentation, infrastructure).
  5. Method 1 — Gabor + SVM (filters, PCA, GridSearch).
  6. Gabor + SVM results.
  7. Method 2 — CNN scratch (architecture).
  8. CNN results + error analysis.
  9. Method 3 — MobileNetV2 (progressive unfreezing).
  10. MobileNetV2 results.
  11. Final comparison (table + bar chart).
  12. Conclusions and future work.

- **How to verify.** Should fit comfortably in an 8-10 minute presentation.

---

## Yeray — MobileNetV2 + comparison

Your work depends on Toni's T1+T2. In the meantime you can get Y1, Y2 and Y4 ready (none of them need `phases/`).

### Task Y1 — `net/backbone/mobilenet.py`

- **What.** Helper that loads the pretrained backbone.
- **How.** A single function:

```python
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def load_pretrained_mobilenet_v2():
    return mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
```

### Task Y2 — `net/networks/mobilenetv2.py` → `MobileNetV2Classifier`

- **What.** Wrapper with a custom head and a `set_phase` method.
- **Goal.**
  - `__init__(num_classes, pretrained=True)`: load backbone, replace the last `Linear(1280, 1000)` with `Linear(1280, num_classes)`.
  - `forward(x)`: standard.
  - `set_phase(unfreeze_from)`:
    - `None` → backbone frozen, only head trains.
    - `int k > 0` → `model.features[k:]` unfrozen.
    - `0` → everything unfrozen.
- **How.**

```python
import torch.nn as nn
from net.backbone.mobilenet import load_pretrained_mobilenet_v2

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        backbone = load_pretrained_mobilenet_v2() if pretrained else \
                   __import__("torchvision").models.mobilenet_v2(weights=None)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)

    def set_phase(self, unfreeze_from):
        for p in self.model.features.parameters():
            p.requires_grad = (unfreeze_from == 0)
        if isinstance(unfreeze_from, int) and unfreeze_from > 0:
            for layer in self.model.features[unfreeze_from:]:
                for p in layer.parameters():
                    p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True
```

### Task Y3 — `run_mobilenetv2` in [main.py](main.py)

- **What.** Replace the `NotImplementedError` in `run_mobilenetv2` (lines 150-154).
- **Why.** Implement the 3 phases (head → partial → full) according to `cfg.mobilenet_phases` already defined in [doc/default.json](doc/default.json).
- **How.** Once Toni's `train_one_ep` and `infer_one_ep` are ready:

```python
def run_mobilenetv2(cfg, run_dir, device):
    import torch
    from dataset.loaders import get_loaders
    from utility.get_fresh_model import get_fresh_model
    from utility.save_checkpoint import save_checkpoint
    from phases.train import train_one_ep
    from phases.infer import infer_one_ep

    train_loader, val_loader, test_loader, classes = get_loaders(
        data_dir=cfg.data_dir,
        subjects_train=cfg.subjects_train, subjects_test=cfg.subjects_test,
        include_extra=cfg.include_extra,
        img_size=cfg.img_size_mobile, batch_size=cfg.batch_size,
        val_split=cfg.val_split, seed=cfg.seed,
        num_workers=cfg.num_workers, mobilenet_norm=True,
    )
    model, loss_fn, _ = get_fresh_model("mobilenetv2", cfg, device, num_classes=len(classes))
    history = []
    for phase in cfg.mobilenet_phases:
        model.set_phase(phase.unfreeze_from)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=phase.lr,
        )
        for epoch in range(1, phase.epochs + 1):
            tr = train_one_ep(model, train_loader, optimizer, loss_fn, device)
            va = infer_one_ep(model, val_loader,  loss_fn, device, validation=True)
            history.append({"phase": phase.name, "epoch": epoch,
                            **{f"train_{k}": v for k, v in tr.items() if k != "y_true"},
                            **{f"val_{k}":   v for k, v in va.items() if k != "y_true"}})
            print(f"[{phase.name} ep {epoch:02d}] "
                  f"train {tr['acc']:.3f} | val {va['acc']:.3f}")
        save_checkpoint(run_dir / "models" / f"{phase.name}.pt", model, optimizer,
                        phase=phase.name, val_acc=va["acc"])
    test = infer_one_ep(model, test_loader, loss_fn, device, validation=False)
    return {"model": "mobilenetv2", "num_classes": len(classes), "classes": classes,
            "test": test, "history": history}
```

- **How to verify.** In Colab: `python main.py --model mobilenetv2 --batch_size 32`. Target: ≥85% test accuracy (reference paper hits 91.8%).

### Task Y4 — `comparison.py`

- **What.** Create `comparison.py` in the repo root.
- **Why.** Final table and figures for the slides.
- **How.** Sketch:

```python
# comparison.py
import json
from pathlib import Path
from metrics.confusion_matrix import confusion_matrix
from visual.plot import plot_confusion_matrix

MODELS = ["gabor_svm", "cnn_scratch", "mobilenetv2"]

def latest_run(model):
    runs = sorted(Path("runs").glob(f"{model}_*"))
    return runs[-1] if runs else None

rows = []
for m in MODELS:
    run = latest_run(m)
    if run is None: continue
    with open(run / "scores" / "results.json") as f:
        r = json.load(f)
    classes = r["classes"]
    test = r["test"]
    cm = confusion_matrix(test["y_true"], test["y_pred"], len(classes))
    Path("figures").mkdir(exist_ok=True)
    plot_confusion_matrix(cm, classes, f"figures/confusion_{m}.png")
    rows.append({"model": m, "acc": test["acc"],
                 "precision_macro": test.get("precision_macro"),
                 "recall_macro":    test.get("recall_macro"),
                 "f1_macro":        test.get("f1_macro")})

print(f"{'Model':<14}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
for r in rows:
    print(f"{r['model']:<14}{r['acc']:>8.3f}{r['precision_macro']:>8.3f}"
          f"{r['recall_macro']:>8.3f}{r['f1_macro']:>8.3f}")
```

- Add a `matplotlib` bar chart with the 4 columns and save it as `figures/comparison_bar.png`.

---

## Shared milestones

- **End of week 2.** All three of `runs/gabor_svm_*/scores/results.json`, `runs/cnn_scratch_*/scores/results.json`, `runs/mobilenetv2_*/scores/results.json` exist (any value — the point is that all three pipelines reach the finish line).
- **End of week 3.** All four PNGs exist: `figures/confusion_gabor_svm.png`, `figures/confusion_cnn_scratch.png`, `figures/confusion_mobilenetv2.png`, `figures/comparison_bar.png`. Plus `presentation/slides.pdf`.

---

## Mini-FAQ

**Where do I run `main.py`?** Locally only to iterate on the code (CPU is slow but enough to validate). For real training, **Colab** (free T4 GPU): `git clone` + `pip install -r doc/requirements.txt` + symlink to the Drive dataset. See [README.md](README.md) for the exact recipe.

**How do I see my batch shape?** Run `python main.py --model dummy`. It prints `(B, 3, 64, 64)` and the split counts.

**What if other teammates haven't finished their part?** Imports in [`utility/get_fresh_model.py`](utility/get_fresh_model.py) are lazy: each model is imported inside its own `if` branch, so another teammate's `NotImplementedError` will never block you.

**What do I do with the `NotImplementedError`s?** Replace them with the implementation. The function docstring is the spec.

**Where do auxiliary helpers go?** Reusable classes in [`net/utility/`](net/utility/), custom losses in [`net/losses/`](net/losses/), plotting helpers in [`visual/`](visual/), checkpoint helpers in [`utility/`](utility/).

**Do I have to hardcode 24 classes?** No. Always `num_classes = len(classes)` where `classes` comes from `get_loaders(...)`.
