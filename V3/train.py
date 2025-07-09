def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x_emg, x_wave, y in loader:
        x_emg, x_wave, y = x_emg.to(device), x_wave.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x_emg, x_wave)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_emg.size(0)
    return total_loss / len(loader.dataset)