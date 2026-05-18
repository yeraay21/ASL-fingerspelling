"""Plotting utilities.

OWNER: Corentin.

Expected interface (consumed by comparison.py and the slides):

    def plot_history(history: list[dict], out_path: str) -> None:
        '''history items have keys: epoch, train_loss, train_acc, val_loss, val_acc.
           Save a 2-panel figure (loss + accuracy curves) as PNG.'''

    def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_path: str) -> None:
        '''Save a heatmap PNG (seaborn.heatmap with annot=True for small CMs).'''
"""


def plot_history(history, out_path: str) -> None:
    raise NotImplementedError("Corentin: implementar plot_history.")


def plot_confusion_matrix(cm, classes, out_path: str) -> None:
    raise NotImplementedError("Corentin: implementar plot_confusion_matrix.")
