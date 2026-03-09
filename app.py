"""
Emotion Detection - Streamlit App
MobileNetV2 (+ DeepFace if available) · 7 Classes

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import sys

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0a0e1a; }
  #MainMenu, footer, header { visibility: hidden; }
  .card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }
  .model-tag {
    font-family: monospace;
    font-size: 0.7rem;
    color: #6b7280;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "angry":    "#ef4444",
    "disgust":  "#65a30d",
    "fear":     "#a855f7",
    "happy":    "#eab308",
    "sad":      "#3b82f6",
    "surprise": "#f97316",
    "neutral":  "#6b7280",
}
EMOTION_EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "sad": "😢", "surprise": "😲", "neutral": "😐",
}

# ── Check DeepFace availability ───────────────────────────────────────────────
@st.cache_resource
def check_deepface():
    try:
        from deepface import DeepFace
        return True, DeepFace
    except Exception:
        return False, None

# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MobileNetV2 model...")
def load_mobilenet():
    import sys, os
    # Add backend folder to path so model_loader etc. can be imported
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from model_loader import load_model
    return load_model()

# ── Inference ─────────────────────────────────────────────────────────────────
def run_mobilenet(image: Image.Image) -> list[dict]:
    import os, sys
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from face_detection import detect_faces
    from inference import predict_emotion

    model = load_mobilenet()
    faces = detect_faces(image)
    results = []
    for face in faces:
        pred = predict_emotion(model, face["face_image"])
        results.append({
            **pred,
            "bbox": {"x": face["x"], "y": face["y"], "w": face["w"], "h": face["h"]},
        })
    return results

def run_deepface(image: Image.Image) -> list[dict]:
    _, DeepFace = check_deepface()
    if DeepFace is None:
        return []
    img_array = np.array(image)
    try:
        raw = DeepFace.analyze(
            img_path=img_array,
            actions=["emotion"],
            enforce_detection=True,
            detector_backend="opencv",
            silent=True,
        )
    except Exception:
        return []
    results = []
    for r in raw:
        dominant = str(r["dominant_emotion"])
        confidence = round(float(r["emotion"][dominant]) / 100.0, 4)
        region = r.get("region", {})
        results.append({
            "emotion": dominant,
            "confidence": confidence,
            "bbox": {
                "x": int(region.get("x", 0)),
                "y": int(region.get("y", 0)),
                "w": int(region.get("w", 0)),
                "h": int(region.get("h", 0)),
            },
        })
    return results

def annotate_image(image: Image.Image, results: list[dict]) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for r in results:
        bbox = r.get("bbox")
        if not bbox:
            continue
        color = EMOTION_COLORS.get(r["emotion"], "#22c55e")
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        label = f"{r['emotion']} {r['confidence']*100:.0f}%"
        try:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_text[2] - bbox_text[0] + 12
        except Exception:
            text_w = len(label) * 9
        draw.rectangle([x, y - 24, x + text_w, y], fill=color)
        draw.text((x + 6, y - 20), label, fill="#000000", font=font)
    return img

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Emotion Detection")
    st.markdown(
        f'<p class="model-tag">MobileNetV2 · 7 Classes · Python {sys.version_info.major}.{sys.version_info.minor}</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    deepface_ok, _ = check_deepface()

    model_options = ["MobileNetV2 (Fast)"]
    if deepface_ok:
        model_options.append("DeepFace (Accurate)")
    else:
        model_options.append("DeepFace — unavailable on this Python")

    selected = st.radio(
        "**Model Backend**",
        model_options,
        help="MobileNetV2 = fast, custom trained. DeepFace = more accurate, slower.",
    )
    use_deepface = "DeepFace (Accurate)" in selected and deepface_ok

    if not deepface_ok:
        st.caption("⚠️ DeepFace requires Python ≤3.12 + TensorFlow. Not available in this environment.")

    st.divider()

    input_mode = st.radio(
        "**Input Source**",
        ["📁 Upload Image", "📷 Webcam Capture"],
    )

    st.divider()
    st.markdown("""
    <div class="model-tag" style="line-height:1.9">
    Backbone: MobileNetV2<br>
    Classifier: FC(512) → BN → ReLU → FC(7)<br>
    Input: 224×224 · Grayscale→3ch<br>
    Detection: MTCNN / OpenCV<br>
    Trained: 24 epochs · acc 66.9%
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("## Facial Emotion Recognition")
st.markdown(
    f'<span class="model-tag">Active model: {"DeepFace" if use_deepface else "MobileNetV2 (custom)"}</span>',
    unsafe_allow_html=True,
)
st.write("")

col_img, col_results = st.columns([3, 2], gap="large")

image: Image.Image | None = None

with col_img:
    if "Upload" in input_mode:
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        captured = st.camera_input("Take a photo", label_visibility="collapsed")
        if captured:
            image = Image.open(io.BytesIO(captured.getvalue())).convert("RGB")

    if image is not None:
        with st.spinner("Analyzing emotions..."):
            results = run_deepface(image) if use_deepface else run_mobilenet(image)

        annotated = annotate_image(image, results)
        st.image(annotated, use_container_width=True)

        if not results:
            st.warning("⚠️ No faces detected in this image.")
    else:
        st.markdown("""
        <div style="
            aspect-ratio:4/3; background:rgba(255,255,255,0.03);
            border:2px dashed rgba(255,255,255,0.1); border-radius:12px;
            display:flex; align-items:center; justify-content:center;
            color:#4b5563; font-family:monospace; font-size:0.9rem;
        ">📷 No image loaded</div>
        """, unsafe_allow_html=True)
        results = []

# ── Results panel ─────────────────────────────────────────────────────────────
with col_results:
    st.markdown("#### Detection Results")

    if image is not None and results:
        for r in results:
            emotion = r["emotion"]
            conf = r["confidence"]
            color = EMOTION_COLORS.get(emotion, "#6b7280")
            emoji = EMOTION_EMOJI.get(emotion, "❓")

            st.markdown(f"""
            <div class="card" style="border-color:{color}44;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="font-size:1.1rem;font-weight:600;color:{color}">{emoji} {emotion.capitalize()}</span>
                    <span style="font-family:monospace;font-size:0.85rem;color:{color}">{conf*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(conf)

        st.divider()
        st.markdown("#### Summary")
        st.markdown(f"""
        <div class="model-tag" style="line-height:2">
        Model: {"DeepFace" if use_deepface else "MobileNetV2"}<br>
        Faces found: {len(results)}<br>
        Dominant: {results[0]["emotion"].capitalize()}
        </div>
        """, unsafe_allow_html=True)

    elif image is not None:
        st.info("No faces detected.")
    else:
        st.markdown(
            '<div style="color:#4b5563;font-family:monospace;font-size:0.85rem;padding:1rem 0">Results will appear here.</div>',
            unsafe_allow_html=True,
        )
