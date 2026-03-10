import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from preprocessing import extract_faces
from inference import predict

dataset = "evaluation_dataset"

y_true = []
y_pred = []

for label in ["real", "fake"]:

    folder = os.path.join(dataset, label)

    for video in os.listdir(folder):

        video_path = os.path.join(folder, video)

        frames, _ = extract_faces(video_path)

        score, prediction = predict(frames)

        pred_label = 1 if prediction == "Fake" else 0
        true_label = 1 if label == "fake" else 0

        y_pred.append(pred_label)
        y_true.append(true_label)


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy:", round(accuracy*100,2), "%")
print("Precision:", round(precision*100,2), "%")
print("Recall:", round(recall*100,2), "%")
print("F1 Score:", round(f1*100,2), "%")