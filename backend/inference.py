import torch
import numpy as np
from model_loader import model, device

def predict(frames):

    if len(frames) == 0:
        return 0.5, "Unknown"

    batch = torch.stack(frames).to(device)

    with torch.no_grad():
        outputs = torch.sigmoid(model(batch)).squeeze()

    probs = outputs.cpu().numpy()

    # average probability
    score = float(np.mean(probs))

    # majority voting
    fake_votes = np.sum(probs > 0.5)
    real_votes = np.sum(probs <= 0.5)

    prediction = "Fake" if fake_votes > real_votes else "Real"
    print("Frame probabilities:", probs)
    print("Model score:",score)

    return score, prediction