import torch
from model_loader import model, device

def predict(frames):
    if len(frames) == 0:
        return 0.0

    batch = torch.stack(frames).to(device)

    with torch.no_grad():
        outputs = torch.sigmoid(model(batch)).squeeze()
        print("Raw outputs:", outputs)

    score = outputs.mean().item()
    return score