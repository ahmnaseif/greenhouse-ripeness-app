# Greenhouse Crop Ripeness Classifier

An end-to-end AI application that classifies crop images by ripeness stage,
built to support automated greenhouse harvesting.

---

## 1. Problem Statement

Automated greenhouse harvesting robots need a reliable way to decide, from a
camera image alone, whether a crop is ready to be picked. Manual inspection
doesn't scale, and harvesting either too early or too late reduces yield and
quality. This project addresses that by providing an AI model that classifies
a crop image into a ripeness stage (e.g. **unripe**, **ripe**, **overripe /
rotten**) in real time.

## 2. Use Case

The application is designed to plug into a greenhouse automation pipeline:

- A harvesting robot's camera captures an image of a crop.
- The image is sent to this application's `/predict` API.
- The API returns the predicted ripeness class and a confidence score.
- The robot (or a human operator, via the web UI) uses that result to decide
  whether to harvest.

It can also be used standalone by greenhouse staff via the web interface to
spot-check produce.

## 3. Solution Overview

A convolutional neural network (transfer learning on MobileNetV2) is trained
to classify crop images into ripeness categories. The trained model is served
behind a FastAPI application that exposes both a JSON API (for integration
with a robot/automation pipeline) and a simple web UI (for manual use). The
application is containerized with Docker and deployed to a cloud platform.

## 4. Dataset

- **Source:** A public fruit/vegetable ripeness dataset (e.g. from
  [Kaggle](https://www.kaggle.com/) or [Hugging Face Datasets](https://huggingface.co/datasets)
  — search for "fruit ripeness classification" or a crop-specific dataset
  matching your target produce).
- **Structure expected by the training script:**
  ```
  data/
      train/
          ripe/
          unripe/
          rotten/
      val/
          ripe/
          unripe/
          rotten/
  ```
  Rename/organize the downloaded dataset's classes into this folder layout
  before training (an `ImageFolder`-style layout, one subfolder per class).
- **Note:** the exact class names are not hardcoded anywhere in the app —
  they're read from whatever folders exist in the dataset at training time
  and saved alongside the model checkpoint, so this works for 2 classes or 5.

## 5. AI/ML Approach

- **Model:** MobileNetV2 pretrained on ImageNet, fine-tuned via transfer
  learning — the convolutional feature extractor is frozen and only a new
  fully-connected classification head is trained. This keeps training fast
  and effective on a small/medium dataset without a GPU being strictly
  required.
- **Framework:** PyTorch + torchvision.
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

- `training/` — offline model training, run separately (locally or on
  Colab), not part of the deployed container.
- `app/` — the deployed FastAPI service (API + web UI + inference).
- `Dockerfile` — builds the `app/` service into a container image.

## 7. Technology Stack

| Layer          | Technology                          |
|----------------|--------------------------------------|
| Model training | PyTorch, torchvision                 |
| Model          | MobileNetV2 (transfer learning)      |
| API / backend  | FastAPI, Uvicorn                     |
| Web UI         | Jinja2 templates, plain HTML/CSS     |
| Containerization | Docker                             |
| Cloud hosting  | Azure App Service (Web App for Containers) + Azure Container Registry |
| Source control | GitHub                               |

## 8. Local Setup Instructions

### 8.1 Train the model (optional if you already have a checkpoint)

```bash
cd training
pip install -r requirements.txt
python train_model.py --data_dir ./data --epochs 10 --output ../app/model/ripeness_model.pt
```

This produces `app/model/ripeness_model.pt` and `app/model/class_names.json`.

### 8.2 Run the API locally

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Web UI: http://localhost:8000/
- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

If `app/model/ripeness_model.pt` isn't present and you'd rather not train
locally, set the `MODEL_URL` environment variable to a direct download link
(e.g. a Hugging Face Hub file URL) and the app will fetch it automatically on
startup — see `app/model_utils.py`.

## 9. Deployment Details

**Cloud platform:** Microsoft Azure — App Service (Web App for Containers),
backed by Azure Container Registry. Deployed using Azure for Students credit.

1. Create a resource group and container registry:
   ```bash
   az group create --name greenhouse-rg --location eastus
   az acr create --resource-group greenhouse-rg --name <your-acr-name> --sku Basic
   ```
2. Build and push the Docker image to the registry:
   ```bash
   az acr login --name <your-acr-name>
   docker build -t <your-acr-name>.azurecr.io/greenhouse-ripeness:latest .
   docker push <your-acr-name>.azurecr.io/greenhouse-ripeness:latest
   ```
3. Create an App Service plan (Linux, B1 — the smallest tier that supports
   custom containers) and the web app:
   ```bash
   az appservice plan create --name greenhouse-plan --resource-group greenhouse-rg --is-linux --sku B1
   az webapp create --resource-group greenhouse-rg --plan greenhouse-plan \
     --name <your-app-name> --deployment-container-image-name <your-acr-name>.azurecr.io/greenhouse-ripeness:latest
   ```
4. Connect the web app to the registry:
   ```bash
   az acr update -n <your-acr-name> --admin-enabled true
   az acr credential show --name <your-acr-name>
   az webapp config container set --name <your-app-name> --resource-group greenhouse-rg \
     --docker-custom-image-name <your-acr-name>.azurecr.io/greenhouse-ripeness:latest \
     --docker-registry-server-url https://<your-acr-name>.azurecr.io \
     --docker-registry-server-user <username> --docker-registry-server-password <password>
   ```
5. Set the container port and, if the model isn't baked into the image, the
   download URL:
   ```bash
   az webapp config appsettings set --resource-group greenhouse-rg --name <your-app-name> \
     --settings WEBSITES_PORT=8000 MODEL_URL="<link to ripeness_model.pt>"
   ```
6. _Fill in the live URL here once deployed:_ `https://<your-app-name>.azurewebsites.net`

**Note:** delete the resource group after evaluation (`az group delete --name greenhouse-rg`)
to stop consuming Azure credit — App Service bills hourly regardless of traffic.

## 10. API / Web Application Usage

**Web UI:** open the deployed URL in a browser, upload a crop image, click
**Classify**.

**JSON API:**

```bash
curl -X POST "https://<your-app-url>/predict" \
  -F "file=@sample_crop.jpg"
```

Example response:

```json
{
  "predicted_class": "ripe",
  "confidence": 0.9421,
  "probabilities": {
    "unripe": 0.031,
    "ripe": 0.9421,
    "rotten": 0.0269
  }
}
```

**Health check:**

```bash
curl https://<your-app-url>/health
```

## 11. Docker Instructions

Build the image:

```bash
docker build -t greenhouse-ripeness .
```

Run it locally:

```bash
docker run -p 8000:8000 \
  -e MODEL_URL="https://<link-to-your-model-file>" \
  greenhouse-ripeness
```

(Omit `-e MODEL_URL=...` if you baked `app/model/ripeness_model.pt` into the
image before building.)

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
│   └── model/                # trained checkpoint goes here (not in git)
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
