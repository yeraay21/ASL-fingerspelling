"""Matriz de confusión (Toni). Filas = true, columnas = predicted."""

from sklearn.metrics import confusion_matrix as sk_cm


def confusion_matrix(y_true, y_pred, num_classes):
    return sk_cm(y_true, y_pred, labels=list(range(num_classes)))
