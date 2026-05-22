"""Top-1 accuracy (Toni). float((y_true == y_pred).mean())."""

import numpy as np


def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())
