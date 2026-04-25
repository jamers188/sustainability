import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["MPLBACKEND"] = "Agg"

import streamlit as st
from PIL import Image
import numpy as np
import time
import io
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────
#  CORE ENGINE SETUP
# ──────────────────────────────────────────────────────────
MODEL_PATH = "best_model.pt"
GDRIVE_FILE_ID = "1FYO7H9UnLDuw5FwAqVpLSvEnPC1dTmod"

def download_model(file_id: str, dest: str):
    import requests
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = session.get(url, stream=True)
    token = next((v for k, v in r.cookies.items() if "download_warning" in k), None)
    if token: r = session.get(url + f"&confirm={token}", stream=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk: f.write(chunk)

if not (os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000):
    download_model(GDRIVE_FILE_ID, MODEL_PATH)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ──────────────────────────────────────────────────────────
#  PROFESSIONAL UI DESIGN (CSS)
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="WasteLens Pro", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@300;400;600&display=swap');

/* Main Body Styling */
.stApp { background-color: #080808; color: #E0E0E0; font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 6rem !important; }

/* Typography */
.logo { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.05em; color: #FFF; margin-bottom: 0; }
.logo span { color: #22C55E; }
.tagline { color: #555; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.4em; margin-bottom: 3rem; }

/* Dashboard Cards */
.glass-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

/* Custom Button Styling */
.stButton > button {
    width: 100%;
    background: #FFF !important;
    color: #000 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.8rem !important;
    transition: 0.3s all;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stButton > button:hover {
    background: #22C55E !important;
    color: #FFF !important;
    transform: translateY(-2px);
}

/* Metric Display */
.metric-container { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 1rem; }
.metric-item { background: #161616; padding: 1rem; border-radius: 8px; border: 1px solid #222; text-align: center; }
.metric-val { font-family: 'Syne', sans-serif; font-size: 1.4rem; color: #FFF; }
.metric-lbl { font-size: 0.6rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em; }

/* Status Indicators */
.status-bar { padding: 1rem; border-radius: 6px; font-weight: 600; font-size: 0.85rem; display: flex; align-items: center; gap: 10px; margin-top: 1rem; }
.status-rec { background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22C55E; color: #22C55E; }
.status-gen { background: rgba(239, 68, 68, 0.1); border-left: 4px solid #EF4444; color: #EF4444; }
.status-mix { background: rgba(245, 158, 11, 0.1); border-left: 4px solid #F59E0B; color: #F59E0B; }

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  INTERFACE LAYOUT
# ──────────────────────────────────────────────────────────

# Minimalist Header
st.markdown('<h1 class="logo">WASTE<span>LENS</span>.PRO</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Enterprise AI Classification Engine</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<p style='font-weight:600; font-size:0.8rem; color:#888; margin-bottom:1.5rem;'>INPUT SOURCE</p>", unsafe_allow_html=True)
    
    # Modern Tabs
    tab_up, tab_cam = st.tabs(["[ UPLOAD ]", "[ CAMERA ]"])
    img_input = None

    with tab_up:
        file = st.file_uploader("Upload", type=['jpg','png','jpeg'], label_visibility="collapsed")
        if file: img_input = Image.open(file).convert("RGB")
    
    with tab_cam:
        cam_file = st.camera_input("Capture", label_visibility="collapsed")
        if cam_file: img_input = Image.open(cam_file).convert("RGB")

    if img_input:
        st.image(img_input, use_container_width=True)
        analyze = st.button("Initialize Analysis")
    else:
        analyze = False
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    if analyze and img_input:
        with st.spinner("QUANTIZING LAYERS..."):
            results = model.predict(img_input, conf=0.3, iou=0.4, verbose=False)
            res = results[0]
            
            # Data Processing
            counts = {"Recyclable": 0, "General": 0}
            items = []
            
            for b in res.boxes:
                name = res.names[int(b.cls[0])].lower()
                is_rec = any(kw in name for kw in ["can", "glass", "paper", "plastic", "bottle", "metal"])
                cat = "Recyclable" if is_rec else "General"
                counts[cat] += 1
                items.append({"label": name.title(), "cat": cat, "conf": float(b.conf[0])})

            # Display Results
            annotated = res.plot(labels=True, conf=False)
            st.image(annotated, use_container_width=True)

            # Pro Metrics Grid
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-item"><div class="metric-val">{len(res.boxes)}</div><div class="metric-lbl">Total</div></div>
                    <div class="metric-item"><div class="metric-val" style="color:#22C55E">{counts['Recyclable']}</div><div class="metric-lbl">Recyclable</div></div>
                    <div class="metric-item"><div class="metric-val" style="color:#EF4444">{counts['General']}</div><div class="metric-lbl">General</div></div>
                </div>
            """, unsafe_allow_html=True)

            # Logic-driven Verdict
            if len(res.boxes) > 0:
                if counts['General'] == 0:
                    st.markdown('<div class="status-bar status-rec">● SYSTEM VERDICT: FULLY RECYCLABLE</div>', unsafe_allow_html=True)
                elif counts['Recyclable'] == 0:
                    st.markdown('<div class="status-bar status-gen">● SYSTEM VERDICT: GENERAL WASTE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-bar status-mix">● SYSTEM VERDICT: MIXED MATERIALS (SORTING REQUIRED)</div>', unsafe_allow_html=True)
            
            # Itemized List
            st.markdown("<p style='font-size:0.7rem; color:#444; margin-top:2rem; letter-spacing:0.1em;'>DETECTION LOG</p>", unsafe_allow_html=True)
            for item in items:
                dot_col = "#22C55E" if item['cat'] == "Recyclable" else "#EF4444"
                st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; padding:8px 0; border-bottom:1px solid #1a1a1a;">
                        <span><span style="color:{dot_col}">■</span> {item['label']}</span>
                        <span style="color:#666">{item['cat']}</span>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="height:350px; border:1px solid #1a1a1a; border-radius:12px; display:flex; align-items:center; justify-content:center; color:#333;">
                <p style="font-size:0.8rem; letter-spacing:0.2em;">SYSTEM IDLE // AWAITING DATA</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<div style='margin-top:5rem; border-top:1px solid #111; padding-top:1rem; text-align:right; font-size:0.6rem; color:#333; letter-spacing:0.2em;'>WASTELENS PRO ENGINE © 2026</div>", unsafe_allow_html=True)
