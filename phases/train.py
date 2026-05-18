"""One-epoch training loop.

OWNER: Corentin.

Expected interface (consumed by main.run_cnn_scratch):

    def train_one_ep(model, loader, optimizer, loss_fn, device) -> dict:
        '''Train model for one epoch over loader. Returns:
              {"loss": float, "acc": float}   # averaged over the epoch
        '''

Pseudocode:
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return {"loss": total_loss / total, "acc": total_correct / total}
"""


def train_one_ep(model, loader, optimizer, loss_fn, device):
    raise NotImplementedError("Corentin: implementar train_one_ep.")
