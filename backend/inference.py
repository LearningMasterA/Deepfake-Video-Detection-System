import torch
import numpy as np
from model_loader import model, device
from gradcam import generate_gradcam

# def predict(frames):

#     if len(frames) == 0:
#         return 0.5, "Unknown"

#     batch = torch.stack(frames).to(device)

#     with torch.no_grad():
#         outputs = torch.sigmoid(model(batch)).squeeze()

#     probs = outputs.cpu().numpy()

#     # average probability
#     score = float(np.mean(probs))

#     # majority voting
#     fake_votes = np.sum(probs > 0.5)
#     real_votes = np.sum(probs <= 0.5)

#     prediction = "Fake" if fake_votes > real_votes else "Real"
#     print("Frame probabilities:", probs)
#     print("Model score:",score)

#     return score, prediction



def predict(frames, raw_images):

    batch = torch.stack(frames).to(device)

    with torch.no_grad():
        outputs = torch.sigmoid(model(batch)).squeeze()

    probs = outputs.cpu().numpy()

    score = float(np.mean(probs))

    # Generate Grad-CAM for first few frames
    heatmaps = []

    for i in range(min(5, len(frames))):
        heatmap = generate_gradcam(frames[i], raw_images[i])
        heatmaps.append(heatmap)

    prediction = "Real" if score > 0.5 else "Fake"

    return score, prediction, heatmaps