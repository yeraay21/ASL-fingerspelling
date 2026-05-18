# ASL Fingerspelling Detection

Comparativa de tres métodos de clasificación de gestos del alfabeto ASL sobre el dataset Kaggle (5 sujetos):

1. **Gabor + SVM** (baseline clásico).
2. **CNN desde cero** (PyTorch).
3. **MobileNetV2 con progressive unfreezing** (transfer learning).

Proyecto de la asignatura de Deep Learning — autores: Yeray, Toni, Corentin.

## Estructura

Ver `plan.md` para la descripción completa. Resumen:

```
dataset/   loaders y preprocesado (incluye banco de Gabor)
net/       arquitecturas PyTorch (CNN scratch, MobileNetV2)
phases/    train_one_ep / infer_one_ep
metrics/   accuracy, per-class, confusion matrix
utility/   save/load checkpoint, get_fresh_model
visual/    plots y matrices de confusión
runs/      generado automáticamente: runs/{model}_{timestamp}/...
data/raw/  dataset Kaggle (no se commitea)
```

## Instalación

```bash
pip install -r doc/requirements.txt
```

## Datos

Descargar el dataset Kaggle "ASL Fingerspelling Images (RGB & Depth)" con
sujetos 1–5. Estructura esperada (24 clases, J y Z excluidas porque
requieren movimiento):

```
data/raw/fingerspelling-asl/
├── subject-1/{A,B,...,Y}/*.jpg    (24 letras: sin J ni Z)
├── subject-2/...
├── subject-3/...
├── subject-4/...
├── subject-5/...
└── extra/                          (opcional, sólo train)
```

Si los descargas en otra ruta, sobreescribe `data_dir` por CLI o edita
`doc/default.json`.

Split: subjects 1–4 → train (90%) + val (10% stratified, seed 42).
Subject 5 → test (nunca visto durante entrenamiento).

## Entrenamiento

```bash
python main.py --model gabor_svm
python main.py --model cnn_scratch  --epochs 25 --batch_size 64
python main.py --model mobilenetv2  --epochs 25 --batch_size 32
```

O con config:

```bash
python main.py --config doc/default.json --model mobilenetv2
```

Cada ejecución crea `runs/{model}_{timestamp}/` con `models/`, `scores/`,
`logs/`, `outputs/` y `parameters/`.

## Comparación

Tras entrenar los 3 modelos:

```bash
python comparison.py
```

Lee todos los `runs/*/scores/results.json` más recientes y genera tabla
comparativa + `figures/comparison_bar.png`.

## Ejecución en Colab

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone <repo-url>
%cd ASL-fingerspelling
!pip install -r doc/requirements.txt
!ln -s /content/drive/MyDrive/ASL_data data/raw/fingerspelling-asl
!python main.py --model mobilenetv2
```
