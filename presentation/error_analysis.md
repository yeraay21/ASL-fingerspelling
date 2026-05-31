# Analisis de errores - CNN Scratch

**Test accuracy global:** 0.4083 (40.8%)
**Numero de clases:** 24
**Total muestras test:** 24000

## Top confusiones mas frecuentes

| # | True | Predicted | Errores | Notas |
|---|------|-----------|---------|-------|
| 1 | **T** | **A** | 961 |  |
| 2 | **S** | **A** | 599 | Puno cerrado, diferencia sutil del pulgar |
| 3 | **G** | **H** | 514 |  |
| 4 | **L** | **A** | 505 |  |
| 5 | **E** | **A** | 496 |  |
| 6 | **Y** | **A** | 451 |  |
| 7 | **O** | **C** | 415 |  |
| 8 | **I** | **O** | 369 |  |
| 9 | **N** | **P** | 367 |  |
| 10 | **N** | **A** | 357 |  |

## Clases con peor F1-score

| Clase | F1 | Precision | Recall | Support |
|-------|----|-----------|--------|---------|
| **M** | 0.000 | 0.000 | 0.000 | 1000 |
| **N** | 0.008 | 0.267 | 0.004 | 1000 |
| **T** | 0.033 | 0.133 | 0.019 | 1000 |
| **G** | 0.070 | 0.725 | 0.037 | 1000 |
| **S** | 0.135 | 0.120 | 0.155 | 1000 |

## Interpretacion

Las confusiones tipicas en ASL fingerspelling:
1. **M/N**: diferencia en num dedos sobre pulgar, sutil a 64x64.
2. **S/A/E**: variantes de puno cerrado.
3. **U/V/R**: dos dedos extendidos con variaciones.

### Posibles mejoras

- Mayor resolucion (224x224) con transfer learning.
- Data augmentation mas agresiva.
- Attention mechanism para focalizar en dedos.
