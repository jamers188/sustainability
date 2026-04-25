import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["MPLBACKEND"] = "Agg"

import streamlit as st
from PIL import Image
import numpy as np
import time
import io
from ultralytics import YOLO

# ─────────────────────────────────────────
#  MODEL SETUP
# ─────────────────────────────────────────
MODEL_PATH = "best_model.pt"
GDRIVE_FILE_ID = "1FYO7H9UnLDuw5FwAqVpLSvEnPC1dTmod"

def download_model(file_id: str, dest: str):
    import requests
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = session.get(url, stream=True)
    token = None
    for k, v in r.cookies.items():
        if "download_warning" in k:
            token = v
    if token:
        r = session.get(url + f"&confirm={token}", stream=True)
    if "text/html" in r.headers.get("Content-Type", ""):
        import re
        match = re.search(r'confirm=([0-9A-Za-z_\-]+)', r.text)
        if match:
            r = session.get(url + f"&confirm={match.group(1)}", stream=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk: f.write(chunk)

# Initial Model Check
if not (os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000):
    with st.spinner("Initializing AI Engine..."):
        download_model(GDRIVE_FILE_ID, MODEL_PATH)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ─────────────────────────────────────────
#  PAGE CONFIG & THEME
# ─────────────────────────────────────────
st.set_page_config(page_title="WasteLens | AI", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

/* Main Background */
.stApp {
    background-color: #050505;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Hide UI Overlays */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 5rem !important; }

/* Header Styling */
.header-container {
    text-align: center;
    margin-bottom: 3rem;
}
.main-title {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.05em;
    color: #FFFFFF;
    margin-bottom: 0;
}
.highlight { color: #22c55e; }
.subtitle {
    color: #666;
    font-size: 0.9rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
}

/* Glassmorphism Cards */
.custom-card {
    background: #0f0f0f;
    border: 1px solid #1f1f1f;
    border-radius: 24px;
    padding: 2rem;
    transition: all 0.3s ease;
}

/* Action Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px !important;
    background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 0 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.2);
}

/* Result Banners */
.verdict-box {
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 15px;
}
.v-rec { background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; color: #22c55e; }
.v-non { background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; color: #ef4444; }
.v-mix { background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; color: #f59e0b; }

/* Metric Styling */
.stat-card {
    background: #141414;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-val { font-size: 1.5rem; font-weight: 700; color: #fff; }
.stat-lbl { font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: 0.1em; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  LOGIC & HELPERS
# ─────────────────────────────────────────
RECYCLABLE_KEYWORDS = ["can", "glass", "paper", "plastic", "cardboard", "bottle", "metal"]
NON_RECYCLABLE_CLASSES = {"foodwaste", "food", "organic", "food_waste"}

def get_class_info(name):
    n = name.lower()
    is_rec = any(kw in n for kw in RECYCLABLE_KEYWORDS) and n not in NON_RECYCLABLE_CLASSES
    friendly = n.replace("_", " ").replace("waste", "").strip().title()
    return is_rec, friendly

# ─────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────

# Centered Header
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">WASTE<span class="highlight">LENS</span></h1>
        <p class="subtitle">Next-Gen Environmental Classification</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("<p style='color:#fff; font-weight:600; margin-bottom:1rem;'>Capture Image</p>", unsafe_allow_html=True)
    
    input_type = st.tabs(["📤 Upload", "📷 Camera"])
    source_image = None
    
    with input_type[0]:
        uploaded = st.file_uploader("Drop image here", type=["jpg","png","jpeg","webp"], label_visibility="collapsed")
        if uploaded:
            source_image = Image.open(uploaded).convert("RGB")
            
    with input_type[1]:
        cam = st.camera_input("Capture", label_visibility="collapsed")
        if cam:
            source_image = Image.open(cam).convert("RGB")

    if source_image:
        st.image(source_image, use_container_width=True, caption="Source Input")
        analyze_btn = st.button("RUN AI ANALYSIS")
    else:
        analyze_btn = False
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if not analyze_btn and "last_run" not in st.session_state:
        st.markdown("""
            <div style="height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 2px dashed #1f1f1f; border-radius: 24px; color: #333;">
                <p style="font-size: 3rem;">⊕</p>
                <p>Waiting for Input Data...</p>
            </div>
        """, unsafe_allow_html=True)
    
    if analyze_btn:
        with st.spinner("Processing Neural Network..."):
            # Set default thresholds internally for a cleaner UI
            results = model.predict(source_image, conf=0.25, iou=0.45, verbose=False)
            res = results[0]
            
            # Metadata extraction
            boxes = res.boxes
            detections = []
            rec_count, non_rec_count = 0, 0
            
            for b in boxes:
                cls_name = res.names[int(b.cls[0])]
                conf = float(b.conf[0])
                is_rec, friendly = get_class_info(cls_name)
                detections.append({"name": friendly, "is_rec": is_rec, "conf": conf})
                if is_rec: rec_count += 1
                else: non_rec_count += 1

            # Output Generation
            annotated_img = res.plot(labels=True, conf=False) # Hide conf on image for professional look
            st.session_state.last_run = {
                "img": Image.fromarray(annotated_img),
                "dets": detections,
                "rec": rec_count,
                "non": non_rec_count,
                "total": len(boxes)
            }

    if "last_run" in st.session_state:
        data = st.session_state.last_run
        st.image(data["img"], use_container_width=True)
        
        # Dashboard Stats
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="stat-card"><div class="stat-val">{data["total"]}</div><div class="stat-lbl">Detected</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#22c55e">{data["rec"]}</div><div class="stat-lbl">Recyclable</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#ef4444">{data["non"]}</div><div class="stat-lbl">General</div></div>', unsafe_allow_html=True)
        
        # Dynamic Verdict Banner
        if data["total"] == 0:
            st.info("No objects identified.")
        elif data["rec"] > 0 and data["non"] == 0:
            st.markdown('<div class="verdict-box v-rec"><b>✓ Fully Recyclable</b> — Place in Green Bin</div>', unsafe_allow_html=True)
        elif data["non"] > 0 and data["rec"] == 0:
            st.markdown('<div class="verdict-box v-non"><b>✕ Non-Recyclable</b> — Place in General Waste</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-box v-mix"><b>⚠ Mixed Materials</b> — Sorting Required</div>', unsafe_allow_html=True)

        # Breakdown List
        for item in data["dets"]:
            color = "#22c55e" if item["is_rec"] else "#ef4444"
            st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding: 0.5rem 0; border-bottom: 1px solid #111;">
                    <span style="color:#eee; font-size:0.9rem;">{item['name']}</span>
                    <span style="color:{color}; font-size:0.7rem; font-weight:700; text-transform:uppercase;">{ 'Recyclable' if item['is_rec'] else 'Waste' }</span>
                </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("<br><br><p style='text-align:center; color:#222; font-size:0.7rem;'>WASTELENS ENGINE V4.0.2 • ENCRYPTED ANALYSIS</p>", unsafe_allow_html=True)
