from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import uuid
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
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frames,raw_images, image_paths = extract_faces(file_path, request_id, start_time)
    
    try:
        score, prediction, heatmaps, fake_score = predict(frames, raw_images)
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
    "frames": image_paths,
    "heatmaps": heatmap_paths
    
}
    
