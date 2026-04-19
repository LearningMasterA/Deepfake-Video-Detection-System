import cv2
import torch
import os
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_DETECTOR = os.getenv("FACE_DETECTOR", "haar").lower()
USE_IMAGENET_NORMALIZATION = os.getenv("USE_IMAGENET_NORMALIZATION", "0") == "1"
FRAME_STRIDE = max(1, int(os.getenv("FRAME_STRIDE", "5")))
mtcnn = MTCNN(keep_all=True, device=device)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

transform_steps = [
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]

if USE_IMAGENET_NORMALIZATION:
    transform_steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

transform = transforms.Compose(transform_steps)

# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

def extract_faces(video_path, output_prefix="frame", start_time_seconds=0.0, max_faces=15):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Unable to open video: {video_path}. The file may be corrupted or "
            "its codec may not be supported by this OpenCV build."
        )

    if start_time_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_seconds * 1000)

    frames = []
    raw_images = []
    image_paths = []

    # create folder for saving frames
    os.makedirs("static", exist_ok=True)

    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Skip more frames by default for faster video analysis.
        if frame_count % FRAME_STRIDE != 0:
            frame_count += 1
            continue

        print(f"Reading frame: {frame_count}")

        detected_faces = []

        if FACE_DETECTOR == "mtcnn":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = mtcnn.detect(frame_rgb)

            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob is None or prob < 0.90:
                        continue

                    x1, y1, x2, y2 = box.astype(int)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    detected_faces.append((x1, y1, x2, y2))

        if not detected_faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for (x, y, w, h) in haar_faces:
                detected_faces.append((x, y, x + w, y + h))

        for (x1, y1, x2, y2) in detected_faces:

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # save raw image for Grad-CAM
            raw_images.append(face)

            # convert to tensor
            face_tensor = transform(face_rgb)

            frames.append(face_tensor)

            # save image for frontend display
            image_path = f"static/{output_prefix}_frame_{saved_count}.jpg"
            cv2.imwrite(image_path, face)

            image_paths.append("/" + image_path)

            saved_count += 1

            # limit number of faces (important for speed)
            if saved_count >= max_faces:
                break

        frame_count += 1

        if saved_count >= max_faces:
            break

    cap.release()

    print(f"Total faces extracted: {len(frames)}")

    return frames, raw_images, image_paths
