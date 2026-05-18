# Plan de Proyecto — ASL Fingerspelling Detection

> Documento guía para el equipo (Yeray, Toni, Corentin).
> Refleja la decisión del 2026-05-18: refactor completo a la estructura del profesor + migración a PyTorch.

---

## Context

El profesor nos ha pasado una estructura de proyecto bastante modular (estilo PyTorch, con `dataset/`, `net/`, `phases/`, `runs/`, etc.). Hasta ahora teníamos `pipeline.py` plano en la raíz con TF/Keras. Hemos decidido:

1. **Refactor completo** a la estructura recomendada — el profesor valorará alineación.
2. **Migrar de Keras a PyTorch** — la estructura del profesor asume claramente PyTorch (`phases/train.py` con `train_one_ep()` es un loop manual, no `model.fit()`).

Esto implica reescribir `pipeline.py` como módulo `dataset/`, e implementar los modelos DL (CNN scratch + MobileNetV2) en PyTorch. Gabor+SVM se queda con sklearn (es método clásico, no hace falta migrar).

El objetivo final no cambia: tener los 3 modelos entrenados, evaluados en el mismo test set (subject-5) y comparados en un notebook/script final, con slides.

### Entorno de ejecución

- **Desarrollo de código:** local (este repo, con git, archivos `.py` limpios). Todos los push van aquí.
- **Entrenamiento:** Google Colab (GPU T4 gratis). Flujo: `git clone` del repo en Colab, montar Drive con `data/raw/`, ejecutar `python main.py --model X`. Los `runs/` resultantes se descargan o se sincronizan con Drive para que todo el equipo los vea.
- Nada de notebooks Colab para el código de los modelos — solo se usa Colab como GPU remota. La única excepción puede ser `comparison.py` si acaba siendo un notebook ligero para mostrar tablas y gráficos.

---

## Estado actual

| Componente | Estado | Notas |
|---|---|---|
| Elección de tema | OK | ASL Fingerspelling, opción (b) |
| Project proposal (docx) | OK | Entregado, humanizado |
| Plan de proyecto (docx) | OK | Versión inicial, este doc lo actualiza |
| Mensaje al profesor | OK | Enviado |
| `pipeline.py` (Keras) | A reemplazar | Funciona, pero hay que rehacerlo como módulo PyTorch |
| Estructura de repo | Pendiente | Solo `pipeline.py` en raíz; falta todo lo demás |
| Modelos | Pendiente | Ninguno implementado todavía |

---

## Estructura objetivo (adaptada del profesor)

```
ASL_Project/
├── README.md
├── doc/
│   ├── default.json           ← configuración por defecto
│   └── requirements.txt       ← torch, torchvision, sklearn, opencv, etc.
├── params.py                  ← argparse + carga de config
├── main.py                    ← entry point. python main.py --model {gabor_svm|cnn_scratch|mobilenetv2}
├── dataset/
│   ├── loaders.py             ← get_train_loader, get_val_loader, get_test_loader
│   ├── dataset.py             ← ASLDataset(Dataset) con __getitem__
│   └── preprocessing/
│       ├── image_processing.py    ← resize, BGR→RGB, normalize
│       ├── custom_transformations.py ← augmentation (torchvision transforms)
│       └── filters.py         ← Gabor filter bank para el baseline clásico
├── metrics/
│   ├── accuracy.py            ← top-1 accuracy
│   ├── per_class.py           ← precision/recall por letra
│   └── confusion_matrix.py
├── net/
│   ├── networks/
│   │   ├── cnn_scratch.py     ← nn.Module CNN custom
│   │   └── mobilenetv2.py     ← wrapper torchvision con cabeza custom
│   ├── backbone/
│   │   └── mobilenet.py       ← weights pretrained (puede ser solo un import)
│   ├── losses/                ← (vacío de inicio; usamos nn.CrossEntropyLoss)
│   └── utility/               ← bloques o capas custom si hicieran falta
├── utility/
│   ├── save_checkpoint.py
│   ├── load_checkpoint.py
│   └── get_fresh_model.py     ← devuelve (model, loss_fn, optimizer) según --model
├── phases/
│   ├── train.py               ← def train_one_ep(model, loader, optim, loss_fn)
│   └── infer.py               ← def infer_one_ep(model, loader, validation=True)
├── visual/
│   ├── plot.py                ← curvas train/val loss y accuracy
│   └── gradcam.py             ← (opcional) explicabilidad MobileNetV2
├── runs/                      ← se crea automáticamente al ejecutar main.py
│   └── {model}_{timestamp}/
│       ├── models/
│       ├── outputs/
│       ├── scores/            ← JSON con métricas finales
│       ├── logs/
│       └── parameters/
├── data/
│   ├── raw/                   ← dataset Kaggle (subject-1 … subject-5, extra, sample)
│   └── processed/             ← .npy o cache opcional si se quiere precomputar
└── presentation/
    └── slides.{pdf,pptx}
```

**Notas sobre la adaptación:**
- `gabor_svm` no es una red, así que no va en `net/`. Su feature extraction vive en `dataset/preprocessing/filters.py` y el entrenamiento se lanza desde `main.py --model gabor_svm` (sklearn, no necesita `phases/`).
- `runs/` se autogenera por experimento — no hacer commit de su contenido (añadir a `.gitignore`).
- `data/raw/` y `data/processed/` no se commitean (8GB).

---

## Reparto de trabajo

Adaptamos el reparto previo a la nueva estructura. Cada persona "posee" sus archivos.

### Yeray — Infraestructura + MobileNetV2 + Comparison
- `dataset/` completo (`dataset.py`, `loaders.py`, `preprocessing/image_processing.py`, `preprocessing/custom_transformations.py`)
- `params.py` + `main.py` + `doc/default.json` + `doc/requirements.txt`
- `utility/` (save/load checkpoint, get_fresh_model)
- `net/networks/mobilenetv2.py` + `net/backbone/mobilenet.py`
- `comparison.py` (script o notebook que lee `runs/*/scores/` y produce tabla y gráficos comparativos finales)
- `README.md` (cómo entrenar y evaluar)

### Toni — Gabor+SVM + visualización + slides
- `dataset/preprocessing/filters.py` (banco de Gabor: 5 freq × 8 orient = 40 filtros)
- Lógica del modelo Gabor+SVM (lanzada desde `main.py --model gabor_svm`): PCA→200 componentes, `GridSearchCV` sobre `C ∈ {0.1, 1, 10}`, kernel RBF
- `visual/plot.py` (curvas, matrices de confusión)
- Esqueleto de slides en `presentation/`

### Corentin — CNN scratch + phases + metrics + análisis de errores
- `net/networks/cnn_scratch.py` (arquitectura ya acordada: Conv32→Conv64→Conv128 + BN + MaxPool + Dense256+Dropout0.5 + Dense26)
- `phases/train.py` (`train_one_ep`) y `phases/infer.py` (`infer_one_ep`)
- `metrics/` (accuracy, per_class precision/recall, confusion matrix)
- Análisis de errores (qué letras se confunden — escribir en `comparison.py` o sección aparte)

---

## Dependencias entre tareas

```
Yeray (dataset/, utility/, params.py, main.py)
   │
   ├──► Corentin (phases/, metrics/, cnn_scratch) ─┐
   │                                                │
   ├──► Yeray (mobilenetv2) ────────────────────────┤
   │                                                │
   └──► Toni (gabor_svm, filters.py) ───────────────┤
                                                    │
                                                    ▼
                                         Yeray (comparison.py)
                                         Toni (visual/, slides)
                                         Corentin (error analysis)
```

**Crítico:** Yeray debe terminar `dataset/` + `utility/` + `main.py` **antes** del miércoles de la semana 2 para que Corentin y Toni puedan trabajar en paralelo. Si Yeray se atasca, todo se bloquea.

---

## Plan semanal

### Semana 1 — Infraestructura base
- **Yeray (lun-mié):** crear estructura de carpetas, `requirements.txt`, `params.py`, `dataset/dataset.py`, `dataset/loaders.py`, `dataset/preprocessing/image_processing.py` y `custom_transformations.py`. Verificar que carga `subject-1..5` correctamente, devuelve tensores PyTorch normalizados.
- **Yeray (jue-vie):** `main.py` con esqueleto de dispatch por `--model`, `utility/save_checkpoint.py` + `load_checkpoint.py` + `get_fresh_model.py`. Crear el directorio `runs/{model}_{timestamp}/`.
- **Corentin (toda la semana):** prepararse en PyTorch (tutorial básico si no lo conoce). Empezar `metrics/accuracy.py` y `metrics/confusion_matrix.py` con datos dummy.
- **Toni (toda la semana):** `dataset/preprocessing/filters.py` con el banco de Gabor. Probar con un par de imágenes que la extracción de features funciona (output shape correcto, no NaN).

**Hito semana 1:** `python main.py --model dummy` corre, crea `runs/`, y los loaders devuelven batches correctos. Gabor filters extraen features. Métricas funcionan con datos dummy.

### Semana 2 — Modelos
- **Yeray (lun-jue):** `net/networks/mobilenetv2.py` con torchvision MobileNetV2 pretrained, cabeza custom de 26 clases. Implementar las 3 fases de progressive unfreezing en `main.py` (o como función separada en `utility/get_fresh_model.py`):
  - Fase 1: freeze backbone, train head, LR=1e-3, ~5 epochs
  - Fase 2: unfreeze layers ≥140, LR=1e-4, ~10 epochs
  - Fase 3: unfreeze all, LR=1e-5, ~10 epochs
  - Guardar checkpoint y métricas tras cada fase.
- **Yeray (vie):** integrar y entrenar end-to-end. Generar `runs/mobilenetv2_*/scores/` con accuracy final.
- **Corentin (lun-mié):** `phases/train.py` y `phases/infer.py` con el loop train/val genérico. `net/networks/cnn_scratch.py`.
- **Corentin (jue-vie):** entrenar el CNN. Adam lr=1e-3, EarlyStopping (implementar manualmente con patience=5), `runs/cnn_scratch_*/scores/`.
- **Toni (toda la semana):** lógica de Gabor+SVM. Lanzar desde `main.py --model gabor_svm`. Pipeline: extract Gabor features → PCA(200) → GridSearchCV(SVM RBF, C∈{0.1,1,10}). Evaluar y guardar `runs/gabor_svm_*/scores/`.

**Hito semana 2:** los 3 modelos entrenados al menos una vez, con `scores/` final guardado en `runs/`.

### Semana 3 — Comparación y entrega
- **Yeray (lun-mar):** `comparison.py` — leer todos los `runs/*/scores/`, tabla comparativa (accuracy, precision/recall macro, F1), gráfico de barras.
- **Toni (lun-mié):** `visual/plot.py` para curvas train/val. Matrices de confusión de los 3 modelos como PNG en `figures/`.
- **Corentin (lun-mié):** análisis de errores — qué letras se confunden más (ej. M/N, S/A), incluir en `comparison.py` o documento aparte.
- **Todos (jue-vie):** slides de presentación. Distribución sugerida: intro + Gabor+SVM (Toni), CNN scratch + análisis errores (Corentin), MobileNetV2 + comparación (Yeray).

**Hito semana 3:** slides terminadas, repo limpio, README explicando cómo reproducir resultados.

---

## Archivos críticos y dependencias técnicas

**Pipeline de datos:**
- `dataset/dataset.py:ASLDataset` — recibe `subjects: list[str]`, `img_size: int`, `transform`. `__getitem__` devuelve `(tensor, label)`.
- `dataset/loaders.py:get_loaders(img_size=64|224, batch_size=64)` — devuelve `(train_loader, val_loader, test_loader)`. Internamente hace el split: subjects 1–4 para train (con val 10% stratified, seed=42), subject-5 para test.
- Preprocesamiento por modelo:
  - CNN/Gabor: 64×64 grayscale, normalize a [0,1]
  - MobileNetV2: 224×224 RGB, normalize con `torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`

**Augmentation (solo train):**
- `RandomRotation(15)`, `RandomAffine(translate=(0.1,0.1), scale=(0.9,1.1))`, `ColorJitter(brightness=0.2)`. **NO flip horizontal** (letras G, J, Z son asimétricas).

**Entry point unificado:**
```bash
python main.py --model mobilenetv2 --epochs 25 --batch_size 64
python main.py --config doc/default.json
```

**Entregables finales:**
- `runs/gabor_svm_*/scores/results.json`
- `runs/cnn_scratch_*/scores/results.json`
- `runs/mobilenetv2_*/scores/results.json` (con métricas por fase)
- `figures/confusion_*.png` (3 archivos)
- `figures/comparison_bar.png`
- `presentation/slides.pdf`

---

## Verificación end-to-end

Antes de considerar el proyecto terminado, debe poderse abrir un Colab nuevo y ejecutar:

```bash
# En Colab
from google.colab import drive
drive.mount('/content/drive')
!git clone <repo>
%cd ASL-fingerspelling
!pip install -r doc/requirements.txt
# data/raw/ apunta a la copia en Drive
!ln -s /content/drive/MyDrive/ASL_data data/raw
!python main.py --model gabor_svm
!python main.py --model cnn_scratch
!python main.py --model mobilenetv2
!python comparison.py
```

Y obtener: 3 carpetas en `runs/`, una tabla comparativa por consola, y los PNGs en `figures/`.

**Criterios de éxito mínimos (MVP):**
- MobileNetV2 ≥ 85% accuracy en test (subject-5). El paper de referencia llega a 91.8%.
- CNN scratch ≥ 50% (cualquier valor por encima de baseline aleatorio 1/26 ≈ 3.8%, idealmente 50–70%).
- Gabor+SVM funcional aunque sea con accuracy modesta (30–60%) — el punto es contrastar con DL.

**Extendido (si sobra tiempo):**
- GradCAM en MobileNetV2 (`visual/gradcam.py`) para ver qué partes de la mano mira.
- Cross-validation por sujeto (leave-one-subject-out).

---

## Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Yeray se atasca con `dataset/` y bloquea al equipo | Media | Empezar lunes mismo. Si jueves no está → simplificar (usar lo que ya hay en pipeline.py adaptado a tensores). |
| Migración a PyTorch lleva más tiempo del esperado | Alta | Si en mitad de semana 1 vamos justos → considerar mantener Keras para los modelos y solo adoptar el "espíritu" de la estructura (carpetas, runs/, etc.). |
| MobileNetV2 no llega al 85% | Baja | Ajustar epochs por fase, probar otro LR. El paper lo consigue, debería funcionar. |
| Colab desconecta entrenamientos largos (>1h sin actividad) | Alta | Guardar checkpoint tras cada epoch (`utility/save_checkpoint.py`) en Drive. Si se cae, retomar con `load_checkpoint`. Mantener la pestaña activa. |
| Colab gratis no garantiza GPU | Media | Si no asigna T4, esperar unas horas o usar Kaggle Notebooks (también T4 gratis, 30h/semana). |
| Sincronización de `runs/` entre los 3 | Media | Configurar `runs/` para que escriba directamente en una carpeta de Drive compartido (`/content/drive/MyDrive/ASL_runs/`). |

---

## Próximo paso inmediato

Crear la estructura de carpetas vacía + `doc/requirements.txt` + esqueleto de `params.py` y `main.py`. Esto desbloquea a Toni y Corentin para empezar sus tareas en paralelo.
