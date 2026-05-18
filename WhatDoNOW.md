> English version: [WhatDoNOW.en.md](WhatDoNOW.en.md)

# WhatDoNOW — guía operativa por integrante

Hoja de ruta accionable para Toni, Corentin y Yeray. El plan canónico vive en [plan.md](plan.md); este documento sólo responde a la pregunta "¿qué tengo que escribir ahora mismo?".

Cada tarea sigue el mismo patrón: **Qué → Por qué → Objetivo → Cómo → Cómo verificar**.

---

## Estado actual del repo

- Infraestructura PyTorch lista y validada con datos reales: `python main.py --model dummy` arranca, descubre **24 clases** (J y Z excluidas porque requieren movimiento), y construye los loaders (train: 69 794, val: 7 755, test: 24 000 imágenes en `subject-5`).
- Lo que ya está hecho (no se toca): [`dataset/`](dataset/), [`utility/`](utility/), [`params.py`](params.py), [`main.py`](main.py), [`doc/`](doc/).
- Lo que está pendiente: los stubs con `NotImplementedError` repartidos en [`net/`](net/), [`phases/`](phases/), [`metrics/`](metrics/), [`visual/`](visual/) y [`dataset/preprocessing/filters.py`](dataset/preprocessing/filters.py).

## Reparto (actualizado)

| Integrante | Track | Bloqueante para |
|---|---|---|
| **Toni** | CNN scratch + phases + metrics + análisis de errores | Yeray (MobileNet) — **camino crítico** |
| **Corentin** | Gabor + SVM + visual/plot + slides | Nadie en semana 2 |
| **Yeray** | MobileNetV2 + comparison | — |

---

## Convenciones comunes (lectura obligatoria)

1. **Trabajar siempre en `main`.** Hacer `git pull` antes de empezar; commits pequeños y descriptivos.
2. **No tocar** [`dataset/`](dataset/), [`utility/`](utility/), [`params.py`](params.py), [`main.py`](main.py) salvo las funciones explícitamente asignadas. Si necesitas cambiar la infraestructura, abre un canal y lo discutimos.
3. **Las imágenes llegan ya como tensores PyTorch** desde los DataLoader: `(B, 3, 64, 64)` para CNN/Gabor o `(B, 3, 224, 224)` para MobileNet. Normalización hecha. Nadie reimplementa carga.
4. **El test set (`subject-5`) no se toca hasta el final.** Sólo `infer_one_ep(..., validation=False)` lo evalúa una vez al terminar el entrenamiento.
5. **Mismo esquema de salida.** Cada modelo guarda en `runs/{model}_{timestamp}/scores/results.json` con: `{model, num_classes, classes, test: {loss, acc, y_true, y_pred}, history?, best_val_acc?}`.
6. **24 clases, no 26.** `num_classes` se pasa por parámetro desde `len(classes)` en runtime. No lo hardcodeéis.
7. **Imports lazy.** El factory [`utility/get_fresh_model.py`](utility/get_fresh_model.py) importa cada modelo dentro de su rama `if`, así que si un stub de otro compañero no compila, tu modelo sigue funcionando.

---

## Toni — CNN scratch + phases + metrics + análisis de errores

Orden recomendado (las tres primeras son crítico para Yeray):

### Tarea T1 — `phases/train.py` → `train_one_ep`

- **Qué.** Rellenar [`phases/train.py`](phases/train.py).
- **Por qué.** Lo llama `main.run_cnn_scratch` (líneas 122-128 de [main.py](main.py)) y lo va a llamar también el `run_mobilenetv2` de Yeray. Sin esto no entrena nadie.
- **Objetivo.** Función `train_one_ep(model, loader, optimizer, loss_fn, device) -> {"loss": float, "acc": float}` con media por época.
- **Cómo.** El pseudocódigo está en el docstring del archivo. Esquema:

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

- **Cómo verificar.** Una vez tengas también T2 y T3, lanzar `python main.py --model cnn_scratch --epochs 1 --batch_size 64`. Debe imprimir `[ep 01] train loss=… acc=…` sin excepciones.

### Tarea T2 — `phases/infer.py` → `infer_one_ep`

- **Qué.** Rellenar [`phases/infer.py`](phases/infer.py).
- **Por qué.** Igual que T1 pero para val y test. Devuelve `y_true/y_pred` para que luego las metrics y la matriz de confusión los consuman.
- **Objetivo.** `infer_one_ep(model, loader, loss_fn, device, validation=True) -> {"loss", "acc", "y_true", "y_pred"}`. Los dos últimos son listas de `int`.
- **Cómo.**

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

- **Cómo verificar.** Igual que T1.

### Tarea T3 — `net/networks/cnn_scratch.py` → `CNNScratch`

- **Qué.** Rellenar [`net/networks/cnn_scratch.py`](net/networks/cnn_scratch.py).
- **Por qué.** Es uno de los tres modelos a comparar.
- **Objetivo.** Una `nn.Module` que reciba un tensor `(B, 3, 64, 64)` y devuelva logits `(B, num_classes)`. **Sin softmax** (CrossEntropyLoss lo aplica internamente).
- **Cómo.** Arquitectura ya fijada en el plan. Esquema en PyTorch:

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

- **Cómo verificar.** Tras T1+T2+T3: `python main.py --model cnn_scratch --epochs 2 --batch_size 64`. La accuracy de validación debería subir entre epoch 1 y 2 (aunque sea poco) en CPU.

### Tarea T4 — `metrics/`

- **Qué.** Rellenar [`metrics/accuracy.py`](metrics/accuracy.py), [`metrics/per_class.py`](metrics/per_class.py), [`metrics/confusion_matrix.py`](metrics/confusion_matrix.py).
- **Por qué.** Yeray los necesita para `comparison.py` en semana 3.
- **Objetivo.** Wrappers finos sobre `sklearn.metrics`. Sin lógica complicada.
- **Cómo.**

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

- **Cómo verificar.** Tests rápidos con listas dummy en un Python interactivo. `accuracy([0,1,2], [0,1,1])` → `0.6666...`.

### Tarea T5 — Entrenar el CNN end-to-end

- **Qué.** Lanzar el entrenamiento completo en Colab (GPU T4) y subir el `runs/cnn_scratch_*/scores/results.json` al repo (haz pull antes, push después).
- **Por qué.** Es uno de los entregables (criterio mínimo: ≥50% accuracy en test).
- **Cómo.** En Colab tras `pip install -r doc/requirements.txt`:

```bash
python main.py --model cnn_scratch --epochs 25 --batch_size 64 --lr 1e-3
```

- **Cómo verificar.** Aparece `runs/cnn_scratch_*/scores/results.json` con `test.acc ≥ 0.50` ideal. Si te quedas en 30-40% revisa augmentation y patience del early-stop.

### Tarea T6 — Análisis de errores

- **Qué.** Identificar las 3-5 confusiones más comunes (típicamente M/N, S/A, U/V/R).
- **Por qué.** Es parte del análisis comparativo final.
- **Cómo.** Cuando Yeray tenga `comparison.py` (semana 3), añadirle una sección. Si no, escribir un párrafo breve en `presentation/error_analysis.md` con los pares más confundidos según `metrics.confusion_matrix(cm)` del modelo CNN.

---

## Corentin — Gabor + SVM + visual/plot + slides

Track independiente: ninguna otra persona te espera en semana 2.

### Tarea C1 — `dataset/preprocessing/filters.py`

- **Qué.** Rellenar [`dataset/preprocessing/filters.py`](dataset/preprocessing/filters.py).
- **Por qué.** Es el feature extractor del baseline clásico.
- **Objetivo.** Dos funciones puras:
  - `build_gabor_bank(frequencies, orientations) -> list[np.ndarray]`: devuelve `len(frequencies) * orientations` kernels 2D reales (40 en total con la config por defecto: 5 frecuencias × 8 orientaciones).
  - `extract_features(images_gray, filters) -> np.ndarray`: matriz `(N, F)` donde `F = 2 * len(filters)` (mean + std de la respuesta de cada filtro).
- **Cómo.** Sugerencia (no obligatoria) con `skimage` y `scipy`:

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

- **Cómo verificar.** En un notebook rápido o REPL: cargar 5 imágenes 64×64, construir el banco, extraer features; el shape final debe ser `(5, 80)` y sin NaN.

### Tarea C2 — Lógica Gabor+SVM en `main.run_gabor_svm`

- **Qué.** Rellenar el cuerpo de `run_gabor_svm` en [main.py](main.py) (líneas 86-93). Actualmente lanza `NotImplementedError`.
- **Por qué.** Es el entry point del modelo Gabor+SVM.
- **Objetivo.** Pipeline completo: extraer features Gabor → reducir dim → entrenar SVM → evaluar en test → guardar `scores/results.json`.
- **Cómo.** Esquema concreto (no usa DataLoader porque sklearn no batched; iterar el dataset directo a NumPy):

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

- **Cómo verificar.** `python main.py --model gabor_svm` produce `runs/gabor_svm_*/scores/results.json` con `test.acc` (objetivo razonable: 30-60%). Suele tardar varios minutos por el GridSearch.

> Nota: si no has tocado sklearn antes, pide a Yeray 30 min para repasar `PCA + GridSearchCV + SVC`. Ahorra horas de debugging.

### Tarea C3 — `visual/plot.py`

- **Qué.** Rellenar [`visual/plot.py`](visual/plot.py): `plot_history` y `plot_confusion_matrix`.
- **Por qué.** `comparison.py` (Yeray) y las slides los consumen para generar los PNG de `figures/`.
- **Objetivo.** Dos funciones que guardan un PNG en disco. No devuelven nada.
- **Cómo.**

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

- **Cómo verificar.** Llamar con datos dummy genera un PNG legible.

### Tarea C4 — Esqueleto de slides

- **Qué.** Crear `presentation/slides.pdf` (o `.pptx`).
- **Por qué.** Es la entrega final junto con el código.
- **Cómo.** Estructura sugerida (~12 slides):
  1. Portada (proyecto, autores, asignatura).
  2. Problema y dataset (ASL, 24 letras, 5 sujetos, split por sujeto).
  3. Por qué split por sujeto importa (variabilidad de manos).
  4. Pipeline común (preprocesado, augmentation, infraestructura).
  5. Método 1 — Gabor + SVM (filtros, PCA, GridSearch).
  6. Resultados Gabor + SVM.
  7. Método 2 — CNN scratch (arquitectura).
  8. Resultados CNN + análisis de errores.
  9. Método 3 — MobileNetV2 (progressive unfreezing).
  10. Resultados MobileNetV2.
  11. Comparativa final (tabla + barras).
  12. Conclusiones y trabajo futuro.

- **Cómo verificar.** Que la cuenten en 8-10 minutos en la presentación.

---

## Yeray — MobileNetV2 + comparison

Tu trabajo depende de T1+T2 de Toni. Mientras tanto puedes adelantar Y1, Y2 y Y4 (no requieren `phases/`).

### Tarea Y1 — `net/backbone/mobilenet.py`

- **Qué.** Helper que carga el backbone pretrained.
- **Cómo.** Una sola función:

```python
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def load_pretrained_mobilenet_v2():
    return mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
```

### Tarea Y2 — `net/networks/mobilenetv2.py` → `MobileNetV2Classifier`

- **Qué.** Wrapper con cabeza custom y método `set_phase`.
- **Objetivo.**
  - `__init__(num_classes, pretrained=True)`: carga backbone, reemplaza la última `Linear(1280, 1000)` por `Linear(1280, num_classes)`.
  - `forward(x)`: estándar.
  - `set_phase(unfreeze_from)`:
    - `None` → backbone congelado, sólo head entrena.
    - `int k > 0` → `model.features[k:]` descongelados.
    - `0` → todo descongelado.
- **Cómo.**

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

### Tarea Y3 — `run_mobilenetv2` en [main.py](main.py)

- **Qué.** Reemplazar el `NotImplementedError` en `run_mobilenetv2` (líneas 150-154).
- **Por qué.** Implementa las 3 fases (head → partial → full) según `cfg.mobilenet_phases` ya definido en [doc/default.json](doc/default.json).
- **Cómo.** Tras tener `train_one_ep` e `infer_one_ep` de Toni:

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

- **Cómo verificar.** En Colab: `python main.py --model mobilenetv2 --batch_size 32`. Objetivo: ≥85% accuracy en test (paper de referencia llega a 91.8%).

### Tarea Y4 — `comparison.py`

- **Qué.** Crear `comparison.py` en la raíz del repo.
- **Por qué.** Tabla y figuras finales para las slides.
- **Cómo.** Pseudocódigo:

```python
# comparison.py
import json, glob
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

- Añade un `matplotlib` bar chart con las 4 columnas y guarda en `figures/comparison_bar.png`.

---

## Hitos compartidos

- **Fin semana 2.** Existen `runs/gabor_svm_*/scores/results.json`, `runs/cnn_scratch_*/scores/results.json`, `runs/mobilenetv2_*/scores/results.json` (cualquier valor, lo importante es que las 3 pipelines lleguen al final).
- **Fin semana 3.** Existen `figures/confusion_gabor_svm.png`, `figures/confusion_cnn_scratch.png`, `figures/confusion_mobilenetv2.png`, `figures/comparison_bar.png`, y `presentation/slides.pdf`.

---

## Mini-FAQ

**¿Dónde corro `main.py`?** En local sólo para iterar el código (CPU es lento pero suficiente para validar). Para entrenar de verdad, **Colab** (GPU T4 gratis): `git clone` + `pip install -r doc/requirements.txt` + simlink al dataset en Drive. Ver [README.md](README.md) para la receta exacta.

**¿Cómo veo qué shape tiene mi batch?** Lanza `python main.py --model dummy`. Imprime `(B, 3, 64, 64)` y el split count.

**¿Y si los otros aún no han acabado su parte?** Los imports en [`utility/get_fresh_model.py`](utility/get_fresh_model.py) son lazy: cada modelo se importa sólo dentro de su rama `if`, así que el `NotImplementedError` de un compañero no te bloquea.

**¿Qué hago con los `NotImplementedError`?** Reemplazarlos por la implementación. El docstring de cada función es la spec.

**¿Dónde meto código auxiliar?** Clases reutilizables en [`net/utility/`](net/utility/), losses custom en [`net/losses/`](net/losses/), funciones de plot en [`visual/`](visual/), helpers de checkpoint en [`utility/`](utility/).

**¿Tengo que hardcodear 24 clases?** No. Siempre `num_classes = len(classes)` con `classes` viniendo de `get_loaders(...)`.
