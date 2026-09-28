import torch
import torch.nn.functional as F

def calculate_loss(model, x, y, vocab_size):
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    return loss

def train_step(model, x, y, optimizer, vocab_size):
    optimizer.zero_grad()

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    loss.backward()
    optimizer.step()

    return loss.item()

def train_model(model, loader, optimizer, vocab_size, device, epochs=1):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            loss = train_step(model, x, y, optimizer, vocab_size)
            total_loss += loss

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")
