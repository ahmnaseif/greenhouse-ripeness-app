# Greenhouse Crop Ripeness Classifier

An end-to-end AI application that classifies crop images by ripeness stage,
built to support automated greenhouse harvesting.

**Live demo:** https://greenhouse-ripeness-app.onrender.com

---

## 1. Problem Statement

Automated greenhouse harvesting robots need a reliable way to decide, from a
camera image alone, whether a crop is ready to be picked. Manual inspection
doesn't scale, and harvesting either too early or too late reduces yield and
quality. This project addresses that by providing an AI model that classifies
a crop image as **fresh** (ripe/harvestable) or **rotten** (spoiled/past
harvest) for a given fruit type, in real time.

## 2. Use Case

The application is designed to plug into a greenhouse automation pipeline:

- A harvesting robot's camera captures an image of a crop.
- The image is sent to this application's `/predict` API.
- The API returns the predicted class (e.g. `freshapples`, `rottenbanana`)
  and a confidence score.
- The robot (or a human operator, via the web UI) uses that result to decide
  whether to harvest.

It can also be used standalone by greenhouse staff via the web interface to
spot-check produce.

## 3. Solution Overview

A convolutional neural network (transfer learning on MobileNetV2) is trained
to classify crop images into freshness/ripeness categories. The trained model
is served behind a FastAPI application that exposes both a JSON API (for
integration with a robot/automation pipeline) and a simple web UI (for manual
use). The application is containerized with Docker and deployed to Render.

## 4. Dataset

- **Source:** [Fruits fresh and rotten for classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) (Kaggle).
- **Classes (6):** `freshapples`, `freshbanana`, `freshoranges`,
  `rottenapples`, `rottenbanana`, `rottenoranges` — three fruit types, each
  labeled fresh (ripe/harvestable) or rotten (spoiled).
- **Structure used for training:**
  ```
  dataset/
      train/
          freshapples/
          freshbanana/
          freshoranges/
          rottenapples/
          rottenbanana/
          rottenoranges/
      val/
          freshapples/
          freshbanana/
          freshoranges/
          rottenapples/
          rottenbanana/
          rottenoranges/
  ```
  (The dataset ships with `train`/`test` folders on Kaggle — `test` was used
  as the `val` split here.)
- **Note:** class names aren't hardcoded anywhere in the app — they're read
  from whatever folders exist in the dataset at training time and saved
  alongside the model checkpoint, so swapping in a different dataset/class
  set requires no code changes.

## 5. AI/ML Approach

- **Model:** MobileNetV2 pretrained on ImageNet, fine-tuned via transfer
  learning — the convolutional feature extractor is frozen and only a new
  fully-connected classification head is trained. This keeps training fast
  and effective on a small/medium dataset without a GPU being strictly
  required.
- **Framework:** PyTorch + torchvision.
- **Training environment:** Google Colab (free T4 GPU).
- **Training script:** [`training/train_model.py`](training/train_model.py)
  handles data loading, augmentation, training, validation, and saving the
  best checkpoint (model weights + class names).
- **Inference:** [`app/model_utils.py`](app/model_utils.py) loads the
  checkpoint and runs preprocessing + prediction on uploaded images.

## 6. Application Architecture

```
┌─────────────┐      image upload       ┌────────────────────┐
│   Client     │ ───────────────────────▶│   FastAPI service   │
│ (browser /   │                          │  ─ /predict (JSON)  │
│  robot API)  │◀──────────────────────── │  ─ /predict-ui (HTML)│
└─────────────┘   JSON / rendered page   │  ─ /health           │
                                          └─────────┬────────────┘
                                                     │ loads at startup
                                                     ▼
                                          ┌────────────────────┐
                                          │ Trained MobileNetV2 │
                                          │ checkpoint (.pt)    │
                                          └────────────────────┘
```

- `training/` — offline model training, run separately (on Colab), not part
  of the deployed container.
- `app/` — the deployed FastAPI service (API + web UI + inference), including
  the trained checkpoint at `app/model/ripeness_model.pt`.
- `Dockerfile` — builds the `app/` service into a container image, built and
  run directly by Render from this GitHub repo.

## 7. Technology Stack

| Layer          | Technology                          |
|----------------|--------------------------------------|
| Model training | PyTorch, torchvision, Google Colab (GPU) |
| Model          | MobileNetV2 (transfer learning)      |
| API / backend  | FastAPI, Uvicorn                     |
| Web UI         | Jinja2 templates, plain HTML/CSS     |
| Containerization | Docker                             |
| Cloud hosting  | Render.com (Web Service, Docker runtime) |
| Source control | GitHub                               |

## 8. Local Setup Instructions

### 8.1 Train the model (optional — a trained checkpoint is already included)

```bash
cd training
pip install -r requirements.txt
python train_model.py --data_dir /path/to/dataset --epochs 10 --output ../app/model/ripeness_model.pt
```

`dataset/` must contain `train/` and `val/` subfolders, each with one
subfolder per class (see Section 4). This produces
`app/model/ripeness_model.pt` and `app/model/class_names.json`.

### 8.2 Run the API locally

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Web UI: http://localhost:8000/
- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

The trained checkpoint is committed at `app/model/ripeness_model.pt`, so no
extra setup is needed to run this locally. If you ever remove it, you can set
a `MODEL_URL` environment variable instead and the app will download it
automatically on startup — see `app/model_utils.py`.

## 9. Deployment Details

**Cloud platform:** [Render](https://render.com) — Web Service, deployed
directly from this GitHub repository using the included `Dockerfile`.

1. Pushed the full project (including the trained `ripeness_model.pt`
   checkpoint) to a public GitHub repository.
2. On Render: **New → Web Service** → connected the GitHub repository.
   Render auto-detected the Dockerfile.
3. Configuration used:
   - **Environment:** Docker
   - **Branch:** `main`
   - **Instance type:** Free
   - No environment variables needed (`MODEL_URL` isn't required since the
     model is baked into the image).
4. Render builds the Docker image and deploys it automatically on every push
   to `main`.
5. **Live URL:** https://greenhouse-ripeness-app.onrender.com

**Note:** Render's free tier spins the service down after ~15 minutes of
inactivity. The first request after idling takes 30–60 seconds to respond
(cold start) while the container restarts — this is expected behavior, not a
fault.

## 10. API / Web Application Usage

**Web UI:** open https://greenhouse-ripeness-app.onrender.com in a browser,
upload a crop image, click **Classify**.

**JSON API:**

```bash
curl -X POST "https://greenhouse-ripeness-app.onrender.com/predict" \
  -F "file=@sample_crop.jpg"
```

Example response:

```json
{
  "predicted_class": "freshapples",
  "confidence": 0.9421,
  "probabilities": {
    "freshapples": 0.9421,
    "freshbanana": 0.0031,
    "freshoranges": 0.0058,
    "rottenapples": 0.0269,
    "rottenbanana": 0.0102,
    "rottenoranges": 0.0119
  }
}
```

**Health check:**

```bash
curl https://greenhouse-ripeness-app.onrender.com/health
```

## 11. Docker Instructions

Build the image:

```bash
docker build -t greenhouse-ripeness .
```

Run it locally:

```bash
docker run -p 8000:8000 greenhouse-ripeness
```


Visit http://localhost:8000/ once the container is running.

---

## Project Structure

```
greenhouse-ripeness-app/
├── training/
│   ├── train_model.py       # model training script
│   └── requirements.txt
├── app/
│   ├── main.py               # FastAPI app (routes)
│   ├── model_utils.py        # model loading + inference
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html        # web UI
│   ├── static/
│   │   └── style.css
│   └── model/
│       └── ripeness_model.pt # trained checkpoint (committed)
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Future Work

- Fine-tune on greenhouse-specific imagery collected from the actual
  harvesting robot's camera, rather than a generic public dataset.
- Extend the model to also flag disease/defects, not just ripeness.
- Integrate the `/predict` endpoint directly into the robot's control loop.
