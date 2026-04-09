import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from preprocessing import extract_faces
from inference import predict

dataset = "evaluation_dataset"

if not os.path.isdir(dataset):
    raise FileNotFoundError(
        f"Missing evaluation dataset folder: {dataset}. Expected "
        f"{dataset}/real and {dataset}/fake."
    )

y_true = []
y_pred = []

for label in ["real", "fake"]:

    folder = os.path.join(dataset, label)

    for video in os.listdir(folder):

        video_path = os.path.join(folder, video)

        frames, raw_images, _ = extract_faces(video_path)

        try:
            score, prediction, _, fake_score, frame_scores = predict(frames, raw_images)
        except RuntimeError as exc:
            raise RuntimeError(f"Cannot evaluate {video_path}: {exc}") from exc

        if prediction not in {"Fake", "Real"}:
            print(f"Skipping {video_path}: {prediction}")
            continue

        pred_label = 1 if prediction == "Fake" else 0
        true_label = 1 if label == "fake" else 0

        y_pred.append(pred_label)
        y_true.append(true_label)

if len(y_true) == 0:
    raise RuntimeError("No videos were evaluated.")

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("Accuracy:", round(accuracy*100,2), "%")
print("Precision:", round(precision*100,2), "%")
print("Recall:", round(recall*100,2), "%")
print("F1 Score:", round(f1*100,2), "%")
