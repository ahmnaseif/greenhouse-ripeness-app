"""
Utilities for loading the ripeness classification model and running inference.

If the model checkpoint isn't present locally and a MODEL_URL environment
variable is set (e.g. pointing to a Hugging Face Hub file), it is downloaded
on first startup. This lets you keep large model weights out of git.
"""

import io
import os
import urllib.request
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "ripeness_model.pt"

_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _ensure_model_file(model_path: Path) -> None:
    if model_path.exists():
        return

    model_url = os.environ.get("MODEL_URL")
    if not model_url:
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}, and no MODEL_URL is set. "
            "Either run training/train_model.py and copy the output into app/model/, "
            "or set the MODEL_URL environment variable to a direct download link."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model from {model_url} ...")
    urllib.request.urlretrieve(model_url, model_path)
    print(f"Model downloaded to {model_path}")


def load_model(model_path: Path = MODEL_PATH) -> Tuple[nn.Module, List[str]]:
    """Load the trained checkpoint, downloading it first if needed."""
    _ensure_model_file(model_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]

    model = _build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(image).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict(model: nn.Module, class_names: List[str], image_bytes: bytes) -> dict:
    tensor = preprocess_image(image_bytes)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze(0)

    top_idx = int(torch.argmax(probs).item())
    return {
        "predicted_class": class_names[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "probabilities": {
            class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))
        },
    }
