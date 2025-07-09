import torch
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for x_emg, x_wave, y in loader:
            x_emg, x_wave, y = x_emg.to(device), x_wave.to(device), y.to(device)
            logits = model(x_emg, x_wave)
            loss = criterion(logits, y)
            val_loss += loss.item() * x_emg.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    return val_loss / total, acc
