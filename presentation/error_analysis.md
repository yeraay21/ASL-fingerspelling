# Analisis de errores - CNN Scratch

**Test accuracy global:** 0.5492 (54.9%)
**Numero de clases:** 24
**Total muestras test:** 24000

## Top confusiones mas frecuentes

| # | True | Predicted | Errores | Notas |
|---|------|-----------|---------|-------|
| 1 | **T** | **A** | 888 |  |
| 2 | **M** | **A** | 662 |  |
| 3 | **S** | **A** | 581 | Puno cerrado, diferencia sutil del pulgar |
| 4 | **E** | **A** | 533 |  |
| 5 | **N** | **A** | 500 |  |
| 6 | **O** | **C** | 458 |  |
| 7 | **L** | **A** | 416 |  |
| 8 | **G** | **H** | 332 |  |
| 9 | **B** | **A** | 321 |  |
| 10 | **Y** | **A** | 298 |  |

## Clases con peor F1-score

| Clase | F1 | Precision | Recall | Support |
|-------|----|-----------|--------|---------|
| **T** | 0.108 | 0.297 | 0.066 | 1000 |
| **M** | 0.185 | 0.350 | 0.126 | 1000 |
| **A** | 0.225 | 0.129 | 0.892 | 1000 |
| **E** | 0.288 | 0.594 | 0.190 | 1000 |
| **N** | 0.290 | 0.781 | 0.178 | 1000 |

## Interpretacion

Las confusiones tipicas en ASL fingerspelling:
1. **M/N**: diferencia en num dedos sobre pulgar, sutil a 64x64.
2. **S/A/E**: variantes de puno cerrado.
3. **U/V/R**: dos dedos extendidos con variaciones.

### Posibles mejoras

- Mayor resolucion (224x224) con transfer learning.
- Data augmentation mas agresiva.
- Attention mechanism para focalizar en dedos.
