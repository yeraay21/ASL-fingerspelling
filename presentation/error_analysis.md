# Analisis de errores - CNN Scratch

**Test accuracy global:** 1.0000 (100.0%)
**Numero de clases:** 24
**Total muestras test:** 240

## Top confusiones mas frecuentes

| # | True | Predicted | Errores | Notas |
|---|------|-----------|---------|-------|

## Clases con peor F1-score

| Clase | F1 | Precision | Recall | Support |
|-------|----|-----------|--------|---------|
| **A** | 1.000 | 1.000 | 1.000 | 10 |
| **B** | 1.000 | 1.000 | 1.000 | 10 |
| **C** | 1.000 | 1.000 | 1.000 | 10 |
| **D** | 1.000 | 1.000 | 1.000 | 10 |
| **E** | 1.000 | 1.000 | 1.000 | 10 |

## Interpretacion

Las confusiones tipicas en ASL fingerspelling:
1. **M/N**: diferencia en num dedos sobre pulgar, sutil a 64x64.
2. **S/A/E**: variantes de puno cerrado.
3. **U/V/R**: dos dedos extendidos con variaciones.

### Posibles mejoras

- Mayor resolucion (224x224) con transfer learning.
- Data augmentation mas agresiva.
- Attention mechanism para focalizar en dedos.
