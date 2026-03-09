"""
Face detection using MTCNN (preferred) with OpenCV Haar Cascade fallback.
"""

import numpy as np
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(keep_all=True, device="cpu", min_face_size=40)
    USE_MTCNN = True
except ImportError:
    import cv2
    _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _cascade = cv2.CascadeClassifier(_cascade_path)
    USE_MTCNN = False


def detect_faces(image: Image.Image) -> list[dict]:
    """
    Detect faces in a PIL image.
    Returns list of dicts with keys: x, y, w, h, face_image (PIL crop).
    """
    if USE_MTCNN:
        return _detect_mtcnn(image)
    return _detect_cascade(image)


def _detect_mtcnn(image: Image.Image) -> list[dict]:
    boxes, _ = _mtcnn.detect(image)
    if boxes is None:
        return []

    results = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        face_crop = image.crop((x1, y1, x2, y2))
        results.append({
            "x": x1, "y": y1,
            "w": x2 - x1, "h": y2 - y1,
            "face_image": face_crop,
        })
    return results


def _detect_cascade(image: Image.Image) -> list[dict]:
    import cv2
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    rects = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    results = []
    for (x, y, w, h) in rects:
        face_crop = image.crop((x, y, x + w, y + h))
        results.append({
            "x": int(x), "y": int(y),
            "w": int(w), "h": int(h),
            "face_image": face_crop,
        })
    return results
