"""
Model loader for MobileNetV2 emotion classifier.
Loads the trained checkpoint and prepares the model for inference.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
MODEL_PATH = Path(__file__).parent / "best_emotion_model.pth"


def build_model(num_classes: int = 7) -> nn.Module:
    """Build MobileNetV2 with custom classifier matching training architecture."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def load_model(path: str | Path = MODEL_PATH, device: str = "cpu") -> nn.Module:
    """Load the trained model checkpoint."""
    model = build_model()

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print(f"[model_loader] Loaded model from {path}")
    print(f"[model_loader] Training accuracy: {checkpoint.get('accuracy', 'N/A')}")
    print(f"[model_loader] Trained for {checkpoint.get('epoch', 'N/A')} epochs")

    return model
