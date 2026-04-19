from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import uuid
import time
from gradcam import generate_gradcam

from model_loader import get_model_metadata
from preprocessing import extract_faces
from inference import DECISION_THRESHOLD, FAKE_CLASS_INDEX, predict

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"
        # "http://localhost:3000",
        # "http://localhost:8000",
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
STATIC_FOLDER = "static"
MAX_UPLOAD_SIZE = 100 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SEGMENT_COUNT = int(os.getenv("SEGMENT_COUNT", "3"))
MAX_TOTAL_FACES = int(os.getenv("MAX_TOTAL_FACES", "15"))


def cleanup_old_generated_files(max_age_seconds=3600):
    now = time.time()
    os.makedirs(STATIC_FOLDER, exist_ok=True)

    for filename in os.listdir(STATIC_FOLDER):
        if "_frame_" not in filename and "_heatmap_" not in filename:
            continue

        path = os.path.join(STATIC_FOLDER, filename)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
            os.remove(path)


def inspect_video_file(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "OpenCV could not open the uploaded video. The file may be corrupted "
            "or encoded with an unsupported codec."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0:
        raise RuntimeError(
            "The uploaded video could not be parsed correctly. OpenCV could not "
            "read its FPS metadata."
        )

    duration_seconds = float(frame_count / fps)
    return {
        "is_valid": True,
        "duration_seconds": duration_seconds,
        "fps": float(fps),
        "frame_count": int(frame_count),
        "width": width,
        "height": height,
        "size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
    }


def build_segment_start_times(start_time, duration, segment_count=SEGMENT_COUNT):
    if duration <= 0:
        return [start_time]

    anchors = [start_time]
    fractions = [0.35, 0.7]

    for fraction in fractions[: max(0, segment_count - 1)]:
        anchors.append(max(start_time, duration * fraction))

    unique_times = []
    for anchor in anchors:
        anchor = min(anchor, max(0.0, duration - 1.0))
        if all(abs(anchor - existing) > 0.5 for existing in unique_times):
            unique_times.append(anchor)

    return unique_times

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
from fastapi.responses import JSONResponse


@app.post("/predict")
async def detect_deepfake(
    file: UploadFile = File(...),
    start_time: float = Form(0.0),
):
    request_id = uuid.uuid4().hex
    start_time = max(0.0, start_time)
    cleanup_old_generated_files()

    _, extension = os.path.splitext(file.filename or "")
    extension = extension.lower()

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Use mp4, mov, avi, mkv, or webm.",
        )

    file_path = os.path.join(UPLOAD_FOLDER, f"{request_id}{extension}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if os.path.getsize(file_path) > MAX_UPLOAD_SIZE:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Video file is larger than 100 MB.")

    try:
        video_validation = inspect_video_file(file_path)
    except RuntimeError as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    duration_seconds = video_validation["duration_seconds"]
    segment_starts = build_segment_start_times(start_time, duration_seconds)
    faces_per_segment = max(1, MAX_TOTAL_FACES // max(1, len(segment_starts)))

    frames = []
    raw_images = []
    image_paths = []

    try:
        for index, segment_start in enumerate(segment_starts):
            segment_frames, segment_raw_images, segment_image_paths = extract_faces(
                file_path,
                f"{request_id}_seg{index}",
                segment_start,
                faces_per_segment,
            )
            frames.extend(segment_frames)
            raw_images.extend(segment_raw_images)
            image_paths.extend(segment_image_paths)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    try:
        score, prediction, heatmaps, fake_score, frame_scores = predict(frames, raw_images)
    except RuntimeError as exc:
        return JSONResponse(
            status_code=200,
            content={
                "prediction": "Model not ready",
                "confidence": None,
                "confidence_display": "Unavailable - model not loaded",
                "frames": image_paths,
                "start_time": start_time,
                "analyzed_segments": segment_starts,
                "heatmaps": [],
                "heatmaps_display": "Unavailable - model not loaded",
                "error": str(exc),
                "message": (
                    "Confidence score and Grad-CAM heatmaps are unavailable "
                    "until a compatible binary deepfake model is loaded."
                ),
            },
        )

    heatmap_paths = []

    for i, heatmap in enumerate(heatmaps):
      path = f"static/{request_id}_heatmap_{i}.jpg"
      cv2.imwrite(path, heatmap)
      heatmap_paths.append("/" + path) 
    
     

    return {
    "prediction": prediction,
    "confidence": score,
    "fake_score": fake_score,
    "fake_class_index": FAKE_CLASS_INDEX,
    "decision_threshold": DECISION_THRESHOLD,
    "model_metadata": get_model_metadata(),
    "video_validation": video_validation,
    "start_time": start_time,
    "analyzed_segments": segment_starts,
    "frame_scores": frame_scores,
    "frames": image_paths,
    "heatmaps": heatmap_paths
    
}
    
