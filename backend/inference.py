import torch
import numpy as np
import os
from model_loader import checkpoint_metadata, model, device, model_load_error
from gradcam import generate_gradcam

# If predictions are reversed for your checkpoint/dataset, switch this between 0 and 1.
FAKE_CLASS_INDEX = int(
    os.getenv(
        "FAKE_CLASS_INDEX",
        str(checkpoint_metadata.get("fake_class_index", 0)),
    )
)
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.30"))
UNCERTAIN_MARGIN = float(os.getenv("UNCERTAIN_MARGIN", "0.08"))
ENABLE_TTA = os.getenv("ENABLE_TTA", "1") != "0"
MAX_HEATMAPS = max(1, int(os.getenv("MAX_HEATMAPS", "5")))

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
    if model is None:
        raise RuntimeError(model_load_error or "Model could not be loaded.")

    if len(frames) == 0:
        return 0.0, "No face detected", [], 0.0, []

    batch = torch.stack(frames).to(device)

    with torch.no_grad():
        outputs = torch.softmax(model(batch), dim=1)
        fake_probs = outputs[:, FAKE_CLASS_INDEX]

        if ENABLE_TTA:
            flipped_batch = torch.flip(batch, dims=[3])
            flipped_outputs = torch.softmax(model(flipped_batch), dim=1)
            fake_probs = (fake_probs + flipped_outputs[:, FAKE_CLASS_INDEX]) / 2.0

    fake_probs = fake_probs.cpu().numpy()
    frame_scores = [
        {"frame": index + 1, "fake_score": float(score)}
        for index, score in enumerate(fake_probs)
    ]

    sorted_probs = np.sort(fake_probs)
    top_k = max(1, int(0.3 * len(sorted_probs)))
    tail_mean = float(sorted_probs[-top_k:].mean())
    median_score = float(np.median(fake_probs))
    fake_probability = 0.6 * tail_mean + 0.4 * median_score

    if abs(fake_probability - DECISION_THRESHOLD) <= UNCERTAIN_MARGIN:
        prediction = "Uncertain"
    else:
        prediction = "Fake" if fake_probability >= DECISION_THRESHOLD else "Real"

    if prediction == "Fake":
        confidence = fake_probability
    elif prediction == "Real":
        confidence = 1.0 - fake_probability
    else:
        confidence = 1.0 - min(1.0, abs(fake_probability - DECISION_THRESHOLD) / max(UNCERTAIN_MARGIN, 1e-6))

    heatmaps = []
    if prediction == "Fake":
        top_indices = np.argsort(fake_probs)[-min(MAX_HEATMAPS, len(frames)):]
        top_indices = sorted(int(index) for index in top_indices)
        for i in top_indices:
            heatmap = generate_gradcam(
                frames[i],
                raw_images[i],
                target_class=FAKE_CLASS_INDEX,
            )
            heatmaps.append(heatmap)

    return confidence, prediction, heatmaps, fake_probability, frame_scores
