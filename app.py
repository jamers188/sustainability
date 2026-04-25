import os
import streamlit as st
from PIL import Image
import numpy as np
import time
import io
from ultralytics import YOLO

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
st.set_page_config(page_title="WasteLens", layout="wide")

# FIXED MODEL SETTINGS (no UI sliders anymore)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

MODEL_PATH = "best_model.pt"

# ─────────────────────────────
# STYLE (clean + premium)
# ─────────────────────────────
st.markdown("""
<style>
body { background-color: #0a0a0a; color: white; }

h1 {
    font-weight: 800;
    letter-spacing: -1px;
}

.stButton>button {
    background: #22c55e;
    color: black;
    border-radius: 10px;
    height: 45px;
    font-weight: 700;
}

.det-box {
    background: #111;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    border: 1px solid #222;
}

.tag-green {
    color: #22c55e;
    font-weight: bold;
}

.tag-red {
    color: #ef4444;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ─────────────────────────────
# HEADER
# ─────────────────────────────
st.markdown("# waste<span style='color:#22c55e'>lens</span>", unsafe_allow_html=True)
st.caption("AI-powered waste classification")

# ─────────────────────────────
# LAYOUT
# ─────────────────────────────
col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    image = None
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_column_width=True)

    run = st.button("Analyse Image", use_container_width=True)

with col2:
    if image is None:
        st.info("Upload an image to start")
    elif run:

        with st.spinner("Analyzing..."):
            results = model.predict(
                np.array(image),
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=640,
                verbose=False
            )

        boxes = results[0].boxes
        names = results[0].names

        detections = []
        for b in boxes:
            detections.append({
                "class": names[int(b.cls[0])].lower()
            })

        # Verdict logic
        recyclable_keywords = ["can", "glass", "paper", "plastic", "bottle"]
        rec = [d for d in detections if any(k in d["class"] for k in recyclable_keywords)]
        nonrec = [d for d in detections if d not in rec]

        if not detections:
            verdict = "No Detection"
        elif rec and not nonrec:
            verdict = "Recyclable"
        elif nonrec and not rec:
            verdict = "Non-Recyclable"
        else:
            verdict = "Mixed"

        # Show image
        st.image(results[0].plot(), caption="Detection")

        # Verdict UI
        if verdict == "Recyclable":
            st.success("♻ Recyclable")
        elif verdict == "Non-Recyclable":
            st.error("✕ Non-Recyclable")
        elif verdict == "Mixed":
            st.warning("⚠ Mixed Waste")
        else:
            st.info("No objects detected")

        # Clean item list (NO CONFIDENCE)
        if detections:
            st.markdown("### Items detected")

            for d in detections:
                is_rec = any(k in d["class"] for k in recyclable_keywords)

                tag = "tag-green" if is_rec else "tag-red"
                label = "Recyclable" if is_rec else "Non-Recyclable"

                st.markdown(
                    f"<div class='det-box'>"
                    f"{d['class'].title()} "
                    f"<span class='{tag}'>({label})</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
