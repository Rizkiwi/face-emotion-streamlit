"""
Emotion inference pipeline:
  Face crop → grayscale → 3ch → resize 224x224 → normalize → model → softmax
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model_loader import EMOTION_CLASSES

# Preprocessing matching training pipeline
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict_emotion(model: torch.nn.Module, face_image: Image.Image) -> dict:
    """
    Run emotion prediction on a single cropped face image.
    Returns {"emotion": str, "confidence": float}
    """
    tensor = preprocess(face_image).unsqueeze(0)  # [1, 3, 224, 224]

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    logits = model(tensor)  # [1, 7]
    probs = F.softmax(logits, dim=1)[0]

    confidence, idx = probs.max(0)

    return {
        "emotion": EMOTION_CLASSES[idx.item()],
        "confidence": round(confidence.item(), 4),
    }
