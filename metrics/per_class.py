"""Per-class precision, recall and F1.

OWNER: Toni. Wrap sklearn.metrics.precision_recall_fscore_support
with labels=list(range(num_classes)) and average=None.

    def per_class_metrics(y_true, y_pred, num_classes: int) -> dict:
        # returns {"precision": [..], "recall": [..], "f1": [..], "support": [..]}
"""


def per_class_metrics(y_true, y_pred, num_classes: int):
    raise NotImplementedError("Toni: implementar per_class_metrics.")
