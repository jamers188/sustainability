import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["MPLBACKEND"] = "Agg"

import streamlit as st
from PIL import Image
import numpy as np
import time
import io
import requests
from ultralytics import YOLO

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="WasteLens",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# FIXED MODEL DEFAULTS
# ─────────────────────────────────────────
CONF_THRESH = 0.25
IOU_THRESH = 0.45
SHOW_LABELS = True
SHOW_CONF_IMG = True

# ─────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────
MODEL_PATH = "best_model.pt"
GDRIVE_FILE_ID = "1FYO7H9UnLDuw5FwAqVpLSvEnPC1dTmod"

def download_model(file_id: str, dest: str):
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
            if chunk:
                f.write(chunk)

model_ok = os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000
if not model_ok:
    if GDRIVE_FILE_ID == "PASTE_YOUR_FILE_ID_HERE":
        st.error("Set your GDRIVE_FILE_ID in app.py")
        st.stop()

    with st.spinner("Downloading model weights... first run only"):
        download_model(GDRIVE_FILE_ID, MODEL_PATH)

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        st.error("Download failed — check GDRIVE_FILE_ID and make sure the file is public.")
        st.stop()

# ─────────────────────────────────────────
# UI CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(34,197,94,0.10), transparent 32rem),
        radial-gradient(circle at bottom right, rgba(34,197,94,0.06), transparent 34rem),
        #050706;
    color: #f5f5f5;
}

.block-container {
    padding: 1.8rem 2.2rem 4rem !important;
    max-width: 1350px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(8, 12, 10, 0.96) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}

[data-testid="stSidebar"] * {
    color: #c7d0ca !important;
}

.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #22c55e;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(255,255,255,0.09);
    margin-bottom: 1rem;
}

.hist-item {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 13px;
    padding: 0.8rem 0.85rem;
    margin-bottom: 0.65rem;
    display: grid;
    grid-template-columns: 0.5fr 1fr 1.4fr;
    gap: 0.45rem;
    align-items: center;
    font-size: 0.74rem;
}

.empty-history {
    color: #66736c;
    font-size: 0.82rem;
    line-height: 1.6;
}

/* Header */
.hero {
    padding: 2.4rem 0 2rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    display: flex;
    justify-content: space-between;
    gap: 1.5rem;
    align-items: flex-end;
}

.brand {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4.4rem);
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.075em;
    line-height: 0.9;
}

.brand span {
    color: #22c55e;
}

.subtitle {
    margin-top: 0.85rem;
    color: #8a968f;
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}

.hero-pill {
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.35);
    color: #8ef0ae;
    padding: 0.55rem 0.95rem;
    border-radius: 999px;
    font-size: 0.64rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* Sections */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 800;
    color: #7e8d84;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

.glass-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 1.25rem;
    box-shadow: 0 24px 70px rgba(0,0,0,0.32);
}

/* Upload */
[data-testid="stFileUploadDropzone"] {
    background: rgba(255,255,255,0.035) !important;
    border: 1px dashed rgba(34,197,94,0.35) !important;
    border-radius: 18px !important;
    padding: 1.2rem !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: #22c55e !important;
    background: rgba(34,197,94,0.055) !important;
}

[data-testid="stFileUploadDropzone"] p {
    color: #8b988f !important;
}

/* Radio */
[data-testid="stRadio"] > label {
    display: none !important;
}

[data-testid="stRadio"] div[role="radiogroup"] {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    padding: 0.35rem;
    gap: 0.55rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #22c55e, #86efac) !important;
    color: #031108 !important;
    border: none !important;
    border-radius: 16px !important;
    height: 3.15rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    box-shadow: 0 16px 42px rgba(34,197,94,0.24) !important;
    transition: all 0.18s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 20px 56px rgba(34,197,94,0.32) !important;
}

.stButton > button:disabled {
    background: rgba(255,255,255,0.06) !important;
    color: #56625b !important;
    box-shadow: none !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: rgba(34,197,94,0.08) !important;
    color: #86efac !important;
    border: 1px solid rgba(34,197,94,0.28) !important;
    border-radius: 15px !important;
    height: 2.9rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 800 !important;
}

/* Images */
img {
    border-radius: 20px;
}

/* Await */
.await-panel {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025));
    border: 1px dashed rgba(255,255,255,0.11);
    border-radius: 26px;
    min-height: 430px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #7f8e86;
    text-align: center;
    padding: 3rem;
    box-shadow: inset 0 0 60px rgba(34,197,94,0.035);
}

.await-icon {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.20);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #22c55e;
    font-size: 1.8rem;
    margin-bottom: 1.15rem;
}

.await-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #e6f4eb;
    margin-bottom: 0.6rem;
}

.await-text {
    color: #7a877f;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* Metrics */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.85rem;
    margin: 1.1rem 0 1.2rem;
}

.metric-box {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem 0.85rem;
    text-align: center;
}

.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
}

.metric-lbl {
    margin-top: 0.45rem;
    color: #7b887f;
    font-size: 0.59rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

/* Verdicts */
.verdict {
    border-radius: 20px;
    padding: 1.05rem 1.15rem;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.9rem;
}

.verdict .v-icon {
    width: 43px;
    height: 43px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
}

.verdict .v-text {
    font-family: 'Syne', sans-serif;
    font-size: 0.96rem;
    font-weight: 800;
    letter-spacing: -0.01em;
}

.verdict .v-sub {
    font-size: 0.78rem;
    margin-top: 0.18rem;
}

.verdict-recyclable {
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.06));
    border: 1px solid rgba(34,197,94,0.33);
}

.verdict-recyclable .v-icon {
    background: rgba(34,197,94,0.12);
    color: #86efac;
}

.verdict-recyclable .v-text {
    color: #86efac;
}

.verdict-recyclable .v-sub {
    color: #63c987;
}

.verdict-nonrecyclable {
    background: linear-gradient(135deg, rgba(239,68,68,0.17), rgba(239,68,68,0.055));
    border: 1px solid rgba(239,68,68,0.34);
}

.verdict-nonrecyclable .v-icon {
    background: rgba(239,68,68,0.12);
    color: #fca5a5;
}

.verdict-nonrecyclable .v-text {
    color: #fca5a5;
}

.verdict-nonrecyclable .v-sub {
    color: #e07c7c;
}

.verdict-mixed {
    background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(245,158,11,0.055));
    border: 1px solid rgba(245,158,11,0.35);
}

.verdict-mixed .v-icon {
    background: rgba(245,158,11,0.12);
    color: #fcd34d;
}

.verdict-mixed .v-text {
    color: #fcd34d;
}

.verdict-mixed .v-sub {
    color: #d9a931;
}

.no-detect-box {
    background: rgba(148,163,184,0.08);
    border: 1px solid rgba(148,163,184,0.17);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    color: #b8c0bb;
    font-size: 0.87rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* Detection items */
.det-item {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem 1.05rem;
    margin-bottom: 0.75rem;
}

.det-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.65rem;
}

.det-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 800;
    color: #f4f8f5;
}

.det-conf {
    font-size: 0.75rem;
    color: #9aa69f;
    font-weight: 700;
}

.conf-track {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
    margin-bottom: 0.75rem;
}

.det-tip {
    color: #8b9890;
    font-size: 0.78rem;
    line-height: 1.6;
}

.chip-rec,
.chip-nonrec {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 0.55rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.22rem 0.48rem;
    border-radius: 999px;
    margin-left: 0.5rem;
    vertical-align: middle;
}

.chip-rec {
    background: rgba(34,197,94,0.10);
    color: #86efac;
    border: 1px solid rgba(34,197,94,0.28);
}

.chip-nonrec {
    background: rgba(239,68,68,0.10);
    color: #fca5a5;
    border: 1px solid rgba(239,68,68,0.30);
}

@media (max-width: 900px) {
    .hero {
        align-items: flex-start;
        flex-direction: column;
    }

    .metric-row {
        grid-template-columns: repeat(2, 1fr);
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
RECYCLABLE_KEYWORDS = ["can", "glass", "paper", "plastic", "cardboard", "bottle", "metal"]
NON_RECYCLABLE_CLASSES = {"foodwaste", "food", "organic", "food_waste"}

DISPOSAL_TIPS = {
    "cans": "Rinse before recycling. Crush to save bin space.",
    "glass": "Remove lids and sort by color if your facility requires it.",
    "paperwaste": "Keep dry. Remove staples and any plastic film.",
    "plasticbottles": "Empty, rinse, and check the resin code on the base.",
    "can": "Rinse before recycling. Crush to save bin space.",
    "paper": "Keep dry. Remove any plastic film or tape.",
    "plastic": "Empty and rinse. Check the resin code on the base.",
    "bottle": "Empty, rinse, and check the resin code on the base.",
    "cardboard": "Flatten before recycling. Remove any tape.",
    "metal": "Rinse clean and place in metals recycling.",
}

DEFAULT_TIP = "Seal in a bag and place in the general waste bin."

def build_class_maps(model):
    recyclable = set()
    friendly = {}

    for idx, name in model.names.items():
        n = name.lower()
        friendly[n] = name.replace("_", " ").replace("waste", "").strip().title()

        if any(kw in n for kw in RECYCLABLE_KEYWORDS) and n not in NON_RECYCLABLE_CLASSES:
            recyclable.add(n)

    return recyclable, friendly

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ─────────────────────────────────────────
# SIDEBAR — HISTORY ONLY
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Detection History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        cmap = {
            "recyclable": "#86efac",
            "non-recyclable": "#fca5a5",
            "mixed": "#fcd34d",
            "no-detection": "#94a3b8",
        }

        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            num = len(st.session_state.history) - i
            c = cmap.get(h["verdict"], "#94a3b8")

            st.markdown(
                f'<div class="hist-item">'
                f'<span style="color:#6b756f;font-weight:800;">#{num}</span>'
                f'<span>{h["count"]} obj</span>'
                f'<span style="color:{c};font-size:0.62rem;font-weight:900;letter-spacing:0.1em;">{h["verdict"].upper()}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="empty-history">No scans yet. Your latest detections will appear here.</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div>
        <div class="brand">waste<span>lens</span></div>
        <div class="subtitle">AI-powered waste classification</div>
    </div>
    <div class="hero-pill">YOLO detection system</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Input Source</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Input",
        ["Upload image", "Use camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    source_image = None

    if mode == "Upload image":
        uploaded = st.file_uploader(
            "upload",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded:
            tmp_path = f"/tmp/wastelens_upload_{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getvalue())

            source_image = Image.open(tmp_path).convert("RGB")
            st.session_state["img_path"] = tmp_path

    else:
        st.caption("Allow camera access in your browser, then click the shutter button.")
        cam = st.camera_input("camera", label_visibility="collapsed")

        if cam:
            tmp_path = "/tmp/wastelens_camera.jpg"
            with open(tmp_path, "wb") as f:
                f.write(cam.getvalue())

            source_image = Image.open(tmp_path).convert("RGB")
            st.session_state["img_path"] = tmp_path

    if source_image:
        st.image(source_image, caption="Original image", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    run = st.button(
        "Analyse image",
        disabled=(source_image is None),
        use_container_width=True,
        type="primary",
    )

with right:
    if source_image is None:
        st.markdown(
            '<div class="await-panel">'
            '<div class="await-icon">⌁</div>'
            '<div class="await-title">Awaiting Scan</div>'
            '<div class="await-text">Upload or capture an image on the left,<br>then press Analyse image to classify the waste.</div>'
            '</div>',
            unsafe_allow_html=True
        )

    elif run:
        with st.spinner("Running detection..."):
            t0 = time.time()

            img_path = st.session_state.get("img_path")
            predict_input = img_path if img_path and os.path.exists(img_path) else np.array(source_image)

            results = model.predict(
                predict_input,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                imgsz=640,
                verbose=False,
            )

            elapsed = time.time() - t0

        boxes = results[0].boxes
        names = results[0].names
        n_det = len(boxes)

        detections = [
            {
                "class": names[int(b.cls[0])].lower(),
                "conf": float(b.conf[0]),
            }
            for b in boxes
        ]

        RECYCLABLE_CLASSES, FRIENDLY = build_class_maps(model)

        annotated_pil = Image.fromarray(
            results[0].plot(labels=SHOW_LABELS, conf=SHOW_CONF_IMG)
        )

        rec = [d for d in detections if d["class"] in RECYCLABLE_CLASSES]
        nonrec = [d for d in detections if d["class"] not in RECYCLABLE_CLASSES]

        verdict = (
            "no-detection" if n_det == 0 else
            "recyclable" if rec and not nonrec else
            "non-recyclable" if nonrec and not rec else
            "mixed"
        )

        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")

        st.session_state.last_result = {
            "annotated_bytes": buf.getvalue(),
            "annotated_pil": annotated_pil,
            "detections": detections,
            "verdict": verdict,
            "elapsed": elapsed,
            "n_det": n_det,
            "rec_count": len(rec),
            "nonrec_count": len(nonrec),
        }

        st.session_state.history.append(
            {
                "count": n_det,
                "verdict": verdict,
            }
        )

    r = st.session_state.last_result

    if r and source_image is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(r["annotated_pil"], caption="Detection overlay", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-box"><div class="metric-val">{r["n_det"]}</div><div class="metric-lbl">Detected</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#86efac">{r["rec_count"]}</div><div class="metric-lbl">Recyclable</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#fca5a5">{r["nonrec_count"]}</div><div class="metric-lbl">Non-Recyclable</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#cbd5e1">{r["elapsed"] * 1000:.0f}ms</div><div class="metric-lbl">Inference</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        v = r["verdict"]

        if v == "recyclable":
            st.markdown(
                '<div class="verdict verdict-recyclable">'
                '<div class="v-icon">♻</div>'
                '<div><div class="v-text">Recyclable</div><div class="v-sub">Place the detected item in the recycling bin.</div></div>'
                '</div>',
                unsafe_allow_html=True
            )

        elif v == "non-recyclable":
            st.markdown(
                '<div class="verdict verdict-nonrecyclable">'
                '<div class="v-icon">✕</div>'
                '<div><div class="v-text">Non-Recyclable</div><div class="v-sub">Place the detected item in the general waste bin.</div></div>'
                '</div>',
                unsafe_allow_html=True
            )

        elif v == "mixed":
            st.markdown(
                '<div class="verdict verdict-mixed">'
                '<div class="v-icon">!</div>'
                '<div><div class="v-text">Mixed Waste</div><div class="v-sub">Separate the recyclable and non-recyclable items before disposal.</div></div>'
                '</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                '<div class="no-detect-box">'
                '<strong>No objects detected.</strong><br>'
                'Try using a clearer image with better lighting and make sure the waste item is visible in the frame.'
                '</div>',
                unsafe_allow_html=True
            )

        if r["detections"]:
            st.markdown(
                '<div class="section-label" style="margin-top:1.35rem;">Item Breakdown</div>',
                unsafe_allow_html=True
            )

            RECYCLABLE_CLASSES, FRIENDLY = build_class_maps(model)

            for d in r["detections"]:
                is_rec = d["class"] in RECYCLABLE_CLASSES
                friendly = FRIENDLY.get(d["class"], d["class"].title())
                tip = DISPOSAL_TIPS.get(d["class"], DEFAULT_TIP)

                chip_cls = "chip-rec" if is_rec else "chip-nonrec"
                chip_lbl = "Recyclable" if is_rec else "Non-Recyclable"

                fill_col = "#86efac" if is_rec else "#fca5a5"
                pct = d["conf"] * 100

                st.markdown(
                    f'<div class="det-item">'
                    f'<div class="det-top">'
                    f'<span class="det-name">{friendly}<span class="{chip_cls}">{chip_lbl}</span></span>'
                    f'<span class="det-conf">{pct:.1f}%</span>'
                    f'</div>'
                    f'<div class="conf-track">'
                    f'<div style="height:6px;border-radius:999px;background:{fill_col};width:{pct:.1f}%"></div>'
                    f'</div>'
                    f'<div class="det-tip">{tip}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.download_button(
            "Download annotated image",
            data=r["annotated_bytes"],
            file_name="wastelens_result.png",
            mime="image/png",
            use_container_width=True,
        )
