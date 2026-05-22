"""Precision/recall/F1 por clase (Toni). Wrapper de sklearn precision_recall_fscore_support."""

from sklearn.metrics import precision_recall_fscore_support


def per_class_metrics(y_true, y_pred, num_classes):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)),
        average=None, zero_division=0
    )
    return {"precision": p.tolist(), "recall": r.tolist(),
            "f1": f.tolist(), "support": s.tolist()}
