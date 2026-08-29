"""
FastAPI application serving the greenhouse crop ripeness classifier.

Endpoints:
    GET  /             - HTML upload form (web UI)
    POST /predict-ui   - handles the form upload, renders the result in the HTML page
    POST /predict       - JSON API: upload an image, get back predicted class + confidence
    GET  /health         - health check used by the cloud platform / uptime monitors
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import model_utils

ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request.
    model, class_names = model_utils.load_model()
    ml_state["model"] = model
    ml_state["class_names"] = class_names
    print(f"Model loaded. Classes: {class_names}")
    yield
    ml_state.clear()


app = FastAPI(
    title="Greenhouse Ripeness Classifier",
    description="Classifies crop images by ripeness stage for automated greenhouse harvesting.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml_state}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None, "error": None})


@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    result, error = None, None
    try:
        result = model_utils.predict(ml_state["model"], ml_state["class_names"], image_bytes)
    except Exception as exc:  # noqa: BLE001 - surface any inference error in the UI
        error = str(exc)

    return templates.TemplateResponse(
        request, "index.html", {"result": result, "error": error}
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        result = model_utils.predict(ml_state["model"], ml_state["class_names"], image_bytes)
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)
