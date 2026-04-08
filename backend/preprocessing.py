import cv2
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

def extract_faces(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []
    raw_images = []
    image_paths = []

    # create folder for saving frames
    os.makedirs("static", exist_ok=True)

    # Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # transform for model input
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # read every 5th frame (faster)
        if frame_count % 5 != 0:
            frame_count += 1
            continue

        print(f"Reading frame: {frame_count}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            # save raw image for Grad-CAM
            raw_images.append(face)

            # convert to tensor
            face_tensor = transform(face)

            frames.append(face_tensor)

            # save image for frontend display
            image_path = f"static/frame_{saved_count}.jpg"
            cv2.imwrite(image_path, face)

            image_paths.append("/" + image_path)

            saved_count += 1

            # limit number of faces (important for speed)
            if saved_count >= 15:
                break

        frame_count += 1

        if saved_count >= 15:
            break

    cap.release()

    print(f"Total faces extracted: {len(frames)}")

    return frames, raw_images, image_paths