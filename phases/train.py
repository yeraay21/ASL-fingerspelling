"""Bucle de entrenamiento de una época (Toni). Devuelve {"loss": ..., "acc": ...}."""


def train_one_ep(model, loader, optimizer, loss_fn, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return {"loss": loss_sum / total, "acc": correct / total}
