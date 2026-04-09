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


def cleanup_old_generated_files(max_age_seconds=3600):
    now = time.time()
    os.makedirs(STATIC_FOLDER, exist_ok=True)

    for filename in os.listdir(STATIC_FOLDER):
        if "_frame_" not in filename and "_heatmap_" not in filename:
            continue

        path = os.path.join(STATIC_FOLDER, filename)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
            os.remove(path)

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

    frames,raw_images, image_paths = extract_faces(file_path, request_id, start_time)
    
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
    "start_time": start_time,
    "frame_scores": frame_scores,
    "frames": image_paths,
    "heatmaps": heatmap_paths
    
}
    
