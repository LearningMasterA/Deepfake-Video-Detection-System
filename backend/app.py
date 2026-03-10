from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from preprocessing import extract_faces
from inference import predict

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
async def detect_deepfake(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frames, image_paths = extract_faces(file_path)
    
    score, prediction = predict(frames)

    confidence = score if prediction == "Fake" else 1 - score
    confidence = round(confidence, 4)   

    return {
        "prediction": prediction,
        "confidence": confidence,
        "frames": image_paths
    
    }
    