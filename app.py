"""
Emotion Detection - Streamlit App
MobileNetV2 + DeepFace · Real-Time · 7 Classes

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark background */
  .stApp { background-color: #0a0e1a; }

  /* Hide default Streamlit header */
  #MainMenu, footer, header { visibility: hidden; }

  /* Card style */
  .card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }

  /* Emotion badge */
  .emotion-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 2px;
  }

  /* Model tag */
  .model-tag {
    font-family: monospace;
    font-size: 0.7rem;
    color: #6b7280;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Result row */
  .result-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
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

# ── Model loaders (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MobileNetV2 model...")
def load_mobilenet():
    from model_loader import load_model
    return load_model()

@st.cache_resource(show_spinner="Checking DeepFace...")
def check_deepface():
    try:
        from deepface import DeepFace
        return True, DeepFace
    except ImportError:
        return False, None

# ── Inference helpers ─────────────────────────────────────────────────────────
def run_mobilenet(image: Image.Image) -> list[dict]:
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
    """Draw bounding boxes and labels on image."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    for r in results:
        bbox = r.get("bbox")
        if not bbox:
            continue
        color = EMOTION_COLORS.get(r["emotion"], "#22c55e")
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        # Box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        # Label background + text
        label = f"{r['emotion']} {r['confidence']*100:.0f}%"
        bbox_text = draw.textbbox((0, 0), label, font=font_small)
        text_w = bbox_text[2] - bbox_text[0] + 12
        text_h = 22
        draw.rectangle([x, y - text_h, x + text_w, y], fill=color)
        draw.text((x + 6, y - text_h + 4), label, fill="#000000", font=font_small)

    return img

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Emotion Detection")
    st.markdown('<p class="model-tag">MobileNetV2 · DeepFace · 7 Classes</p>', unsafe_allow_html=True)
    st.divider()

    # Model selector
    deepface_ok, _ = check_deepface()
    model_options = ["MobileNetV2 (Fast)"]
    if deepface_ok:
        model_options.append("DeepFace (Accurate)")
    else:
        model_options.append("DeepFace (not installed)")

    selected = st.radio(
        "**Model Backend**",
        model_options,
        disabled=False,
        help="MobileNetV2 = fast, custom trained. DeepFace = more accurate, slower.",
    )
    use_deepface = "DeepFace" in selected and deepface_ok

    if not deepface_ok:
        st.caption("💡 Install DeepFace: `pip install deepface`")

    st.divider()

    # Input mode
    input_mode = st.radio(
        "**Input Source**",
        ["📁 Upload Image", "📷 Webcam Capture"],
    )

    st.divider()
    st.markdown("""
    <div class="model-tag" style="line-height:1.8">
    Backbone: MobileNetV2<br>
    Classifier: FC(512) → BN → ReLU → FC(7)<br>
    Input: 224×224 · Grayscale→3ch<br>
    Detection: MTCNN / OpenCV
    </div>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## Facial Emotion Recognition")
st.markdown(
    f'<span class="model-tag">Using: {"DeepFace" if use_deepface else "MobileNetV2 (custom)"}</span>',
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

    else:  # Webcam
        captured = st.camera_input("Take a photo", label_visibility="collapsed")
        if captured:
            image = Image.open(io.BytesIO(captured.getvalue())).convert("RGB")

    # Run inference
    if image is not None:
        with st.spinner("Analyzing emotions..."):
            results = run_deepface(image) if use_deepface else run_mobilenet(image)

        annotated = annotate_image(image, results)
        st.image(annotated, use_container_width=True, caption="")

        if not results:
            st.warning("⚠️ No faces detected in this image.")
    else:
        # Placeholder
        st.markdown("""
        <div style="
            aspect-ratio: 4/3;
            background: rgba(255,255,255,0.03);
            border: 2px dashed rgba(255,255,255,0.1);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #4b5563;
            font-family: monospace;
            font-size: 0.9rem;
        ">
            📷 No image loaded
        </div>
        """, unsafe_allow_html=True)
        results = []

# ── Results panel ─────────────────────────────────────────────────────────────
with col_results:
    st.markdown("#### Detection Results")

    if image is not None and results:
        for i, r in enumerate(results):
            emotion = r["emotion"]
            conf = r["confidence"]
            color = EMOTION_COLORS.get(emotion, "#6b7280")
            emoji = EMOTION_EMOJI.get(emotion, "❓")

            with st.container():
                st.markdown(f"""
                <div class="card" style="border-color: {color}33;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:1.1rem; font-weight:600; color:{color}">
                            {emoji} {emotion.capitalize()}
                        </span>
                        <span style="font-family:monospace; font-size:0.85rem; color:{color}">
                            {conf*100:.1f}%
                        </span>
                    </div>
                """, unsafe_allow_html=True)

                st.progress(conf, text="")
                st.markdown("</div>", unsafe_allow_html=True)

        # Summary
        if len(results) > 1:
            st.caption(f"🔍 {len(results)} faces detected")

        # All emotion scores (DeepFace gives full breakdown)
        st.divider()
        st.markdown("#### Model Info")
        st.markdown(f"""
        <div class="model-tag" style="line-height:2">
        Model: {"DeepFace" if use_deepface else "MobileNetV2"}<br>
        Faces found: {len(results)}<br>
        Dominant: {results[0]["emotion"].capitalize() if results else "—"}
        </div>
        """, unsafe_allow_html=True)

    elif image is not None and not results:
        st.info("No faces detected.")
    else:
        st.markdown("""
        <div style="color:#4b5563; font-family:monospace; font-size:0.85rem; padding: 1rem 0;">
            Results will appear here after analyzing an image.
        </div>
        """, unsafe_allow_html=True)
