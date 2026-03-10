import cv2
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_faces(video_path, max_frames=15):
    cap = cv2.VideoCapture(video_path)
    frames = []
    saved_paths = []
    count = 0

    # Clear old extracted images
    output_dir = "static/extracted"
    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_img = Image.fromarray(face_rgb)

            # Save for frontend display
            save_path = os.path.join(output_dir, f"frame_{count}.jpg")
            face_img.save(save_path)

            saved_paths.append("/static/extracted/" + f"frame_{count}.jpg")

            frames.append(transform(face_img))
            break

        count += 1

    cap.release()

    print("Total faces extracted:", len(frames))
    return frames, saved_paths