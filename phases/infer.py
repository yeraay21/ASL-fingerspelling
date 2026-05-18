"""Validation / test inference loop.

OWNER: Toni. CRITICAL PATH — Yeray's MobileNetV2 also depends on this function.

Expected interface (consumed by main.run_cnn_scratch):

    def infer_one_ep(model, loader, loss_fn, device, validation: bool = True) -> dict:
        '''Run inference over loader (no grad). Returns at minimum:
            {"loss": float, "acc": float, "y_true": list[int], "y_pred": list[int]}
           y_true/y_pred enable downstream confusion matrix + per-class metrics.
        '''

Pseudocode:
    model.eval()
    with torch.no_grad():
        ...accumulate loss, count correct, append y_true/y_pred lists...
"""


def infer_one_ep(model, loader, loss_fn, device, validation: bool = True):
    raise NotImplementedError("Toni: implementar infer_one_ep.")
