"""Bucle de inferencia val/test (Toni). Devuelve {"loss", "acc", "y_true", "y_pred"}."""

import torch


def infer_one_ep(model, loader, loss_fn, device, validation=True):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            preds = logits.argmax(1)
            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y).sum().item()
            total += x.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return {"loss": total_loss / total, "acc": total_correct / total,
            "y_true": y_true, "y_pred": y_pred}