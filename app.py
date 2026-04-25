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
#  MODEL DOWNLOAD
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
            if chunk:
                f.write(chunk)

model_ok = os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000
if not model_ok:
    if GDRIVE_FILE_ID == "PASTE_YOUR_FILE_ID_HERE":
        st.error("Set your GDRIVE_FILE_ID in app.py")
        st.stop()

    with st.spinner("Downloading model weights... (~6MB, first run only)"):
        download_model(GDRIVE_FILE_ID, MODEL_PATH)

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        st.error("Download failed — check GDRIVE_FILE_ID and that the file is shared publicly.")
        st.stop()

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="WasteLens",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding: 1.4rem 2rem 5rem !important;
    max-width: 1300px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0a0a !important;
    border-right: 1px solid #1f1f1f !important;
}

[data-testid="stSidebar"] * {
    color: #a0a0a0 !important;
}

/* ── Upload zone ── */
[data-testid="stFileUploadDropzone"] {
    background: #0f0f0f !important;
    border: 2px dashed #22c55e !important;
    border-radius: 14px !important;
    min-height: 120px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    padding: 1.5rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: #4ade80 !important;
    background: #111111 !important;
    box-shadow: 0 0 24px rgba(34,197,94,0.12) !important;
}

[data-testid="stFileUploadDropzone"] p {
    color: #9ca3af !important;
    text-align: center !important;
    font-size: 0.9rem !important;
}

[data-testid="stFileUploadDropzone"] svg {
    color: #22c55e !important;
    fill: #22c55e !important;
    width: 2.1rem !important;
    height: 2.1rem !important;
}

/* ── Analyse button ── */
.stButton > button {
    background: #22c55e !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    height: auto !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 0 24px rgba(34,197,94,0.35) !important;
    opacity: 1 !important;
}

.stButton > button:hover {
    background: #4ade80 !important;
    color: #000 !important;
    box-shadow: 0 0 34px rgba(34,197,94,0.45) !important;
    transform: translateY(-1px);
}

.stButton > button:disabled {
    background: #1a1a1a !important;
    color: #3f3f3f !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #0f0f0f !important;
    color: #22c55e !important;
    border: 1px solid #1a3a24 !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    width: 100% !important;
    height: auto !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}

/* ── Await panel ── */
.await-panel {
    background: #0a0a0a;
    border: 1px dashed #2a2a2a;
    border-radius: 16px;
    min-height: 400px;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 3rem 2.5rem;
    margin-top: 0.2rem;
}

.await-icon {
    font-size: 4rem;
    margin-bottom: 1.2rem;
    color: #22c55e;
    filter: none;
}

.await-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #e5e7eb;
    margin-bottom: 0.7rem;
}

.await-text {
    font-size: 0.92rem;
    color: #9ca3af;
    line-height: 1.7;
}

/* ── Verdict banners ── */
.verdict-recyclable {
    background: linear-gradient(135deg, #0a1f12 0%, #0f2d1a 100%);
    border: 1px solid #166534;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.verdict-recyclable .v-icon {
    font-size: 1.3rem;
}

.verdict-recyclable .v-text {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    color: #22c55e;
}

.verdict-recyclable .v-sub {
    font-size: 0.72rem;
    color: #15803d;
    margin-top: 2px;
}

.verdict-nonrecyclable {
    background: linear-gradient(135deg, #1a0a0a 0%, #2a1010 100%);
    border: 1px solid #7f1d1d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.verdict-nonrecyclable .v-icon {
    font-size: 1.3rem;
}

.verdict-nonrecyclable .v-text {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    color: #ef4444;
}

.verdict-nonrecyclable .v-sub {
    font-size: 0.72rem;
    color: #b91c1c;
    margin-top: 2px;
}

.verdict-mixed {
    background: linear-gradient(135deg, #1a1500 0%, #2a2200 100%);
    border: 1px solid #713f12;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.verdict-mixed .v-icon {
    font-size: 1.3rem;
}

.verdict-mixed .v-text {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    color: #f59e0b;
}

.verdict-mixed .v-sub {
    font-size: 0.72rem;
    color: #b45309;
    margin-top: 2px;
}

/* ── Metrics ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 0.75rem;
    margin-top: 1.2rem;
    margin-bottom: 1.5rem;
}

.metric-box {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 10px;
    padding: 1.4rem 1rem;
    text-align: center;
}

.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: #fff;
    line-height: 1;
}

.metric-lbl {
    font-size: 0.6rem;
    color: #3f3f3f;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 5px;
    font-weight: 500;
}

/* ── Detection items ── */
.det-item {
    background: #111111;
    border: 1px solid #222222;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.15s;
}

.det-item:hover {
    border-color: #2a2a2a;
}

.det-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.det-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #e5e5e5;
}

.det-conf {
    font-size: 0.7rem;
    color: #525252;
    font-weight: 500;
}

.conf-track {
    background: #1a1a1a;
    border-radius: 3px;
    height: 7px;
    margin-bottom: 10px;
}

.conf-track div {
    height: 7px !important;
}

.det-tip {
    font-size: 0.72rem;
    color: #525252;
    line-height: 1.6;
}

.chip-rec {
    display: inline-block;
    font-size: 0.58rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.1em;
    padding: 2px 7px;
    border-radius: 100px;
    margin-left: 8px;
    background: #166534;
    color: #4ade80;
    border: 1px solid #15803d;
    vertical-align: middle;
}

.chip-nonrec {
    display: inline-block;
    font-size: 0.58rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.1em;
    padding: 2px 7px;
    border-radius: 100px;
    margin-left: 8px;
    background: #166534;
    color: #4ade80;
    border: 1px solid #15803d;
    vertical-align: middle;
}

/* ── History ── */
.hist-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #141414;
    font-size: 0.85rem;
}

.hist-item:last-child {
    border-bottom: none;
}

/* ── Sidebar label ── */
.sb-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #22c55e !important;
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 0.7rem;
    margin-bottom: 1rem;
}

/* ── Radio ── */
[data-testid="stRadio"] > label {
    display: none !important;
}

[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 0.5rem !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label span {
    font-size: 0.82rem !important;
    color: #a0a0a0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────
CONF_THRESH = 0.25
IOU_THRESH = 0.45
SHOW_LABELS = True
SHOW_CONF_IMG = True

RECYCLABLE_KEYWORDS = ["can", "glass", "paper", "plastic", "cardboard", "bottle", "metal"]
NON_RECYCLABLE_CLASSES = {"foodwaste", "food", "organic", "food_waste"}

DISPOSAL_TIPS = {
    "cans":           "Rinse before recycling. Crush to save bin space.",
    "glass":          "Remove lids and sort by color if your facility requires it.",
    "paperwaste":     "Keep dry. Remove staples and any plastic film.",
    "plasticbottles": "Empty, rinse, and check the resin code on the base.",
    "can":            "Rinse before recycling. Crush to save bin space.",
    "paper":          "Keep dry. Remove any plastic film or tape.",
    "plastic":        "Empty and rinse. Check the resin code on the base.",
    "bottle":         "Empty, rinse, and check the resin code on the base.",
    "cardboard":      "Flatten before recycling. Remove any tape.",
    "metal":          "Rinse clean and place in metals recycling.",
}

DEFAULT_TIP = "Seal in a bag and place in the general waste bin."

def build_class_maps(model):
    """Dynamically build recyclable set and friendly names from the model's actual class list."""
    recyclable = set()
    friendly = {}

    for idx, name in model.names.items():
        n = name.lower()
        friendly[n] = name.replace("_", " ").replace("waste", "").strip().title()

        if any(kw in n for kw in RECYCLABLE_KEYWORDS) and n not in NON_RECYCLABLE_CLASSES:
            recyclable.add(n)

    return recyclable, friendly

# ─────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-label">
        Detection History
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

        cmap = {
            "recyclable": "#7ed4a0",
            "non-recyclable": "#e07070",
            "mixed": "#d4b86a",
            "no-detection": "#6b7280",
        }

        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            num = len(st.session_state.history) - i
            c = cmap.get(h["verdict"], "#6b7280")

            st.markdown(
                f'<div class="hist-item">'
                f'<span style="color:#4b5563;font-family:Space Mono,monospace;font-size:0.85rem">#{num}</span>'
                f'<span style="color:#9ca3af">{h["count"]} obj</span>'
                f'<span style="color:{c};font-family:Space Mono,monospace;font-size:0.85rem">{h["verdict"].upper()}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown('<p style="color:#6b7280;font-size:0.9rem;line-height:1.6;">No scans yet.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="
    padding: 3rem 0 2.4rem;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 2.6rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
">
    <div>
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            color: #fff;
            letter-spacing: -0.05em;
            line-height: 1;
            margin-bottom: 0.4rem;
        ">waste<span style="color:#22c55e">lens</span></div>
        <div style="
            font-size: 0.7rem;
            color: #404040;
            letter-spacing: 0.25em;
            text-transform: uppercase;
            font-weight: 400;
        ">AI-powered waste classification</div>
    </div>
    <div style="
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #22c55e;
        background: #0a1f10;
        border: 1px solid #14532d;
        padding: 6px 14px;
        border-radius: 100px;
        margin-bottom: 4px;
    ">v4 model</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;
        color:#404040;margin-bottom:1rem;font-family:'Syne',sans-serif;">
        Input source
    </div>""", unsafe_allow_html=True)

    mode = st.radio("Input", ["Upload image", "Use camera"], horizontal=True, label_visibility="collapsed")
    source_image = None

    st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)

    if mode == "Upload image":
        uploaded = st.file_uploader("upload", type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")

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
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.image(source_image, caption="Original image", use_column_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    run = st.button(
        "Analyse image",
        disabled=(source_image is None),
        use_container_width=True,
        type="primary"
    )

with right:
    if source_image is None:
        st.markdown(
            '<div class="await-panel">'
            '<div class="await-icon">&#9707;</div>'
            '<div class="await-title">Awaiting scan</div>'
            '<div class="await-text">Upload or capture an image on the left,<br>then press Analyse image.</div>'
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
                verbose=False
            )

            elapsed = time.time() - t0

        boxes = results[0].boxes
        names = results[0].names
        n_det = len(boxes)

        detections = [
            {
                "class": names[int(b.cls[0])].lower(),
                "conf": float(b.conf[0])
            }
            for b in boxes
        ]

        RECYCLABLE_CLASSES, FRIENDLY = build_class_maps(model)

        annotated_pil = Image.fromarray(results[0].plot(labels=SHOW_LABELS, conf=SHOW_CONF_IMG))

        rec = [d for d in detections if d["class"] in RECYCLABLE_CLASSES]
        nonrec = [d for d in detections if d["class"] not in RECYCLABLE_CLASSES]

        verdict = (
            "no-detection"   if n_det == 0 else
            "recyclable"     if rec and not nonrec else
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

        st.session_state.history.append({
            "count": n_det,
            "verdict": verdict
        })

    r = st.session_state.last_result

    if r and source_image is not None:
        st.image(r["annotated_pil"], caption="Detection overlay", use_column_width=True)

        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-box"><div class="metric-val">{r["n_det"]}</div><div class="metric-lbl">Detected</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#7ed4a0">{r["rec_count"]}</div><div class="metric-lbl">Recyclable</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#e07070">{r["nonrec_count"]}</div><div class="metric-lbl">Non-recyclable</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#9ca3af">{r["elapsed"]*1000:.0f}ms</div><div class="metric-lbl">Inference</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        v = r["verdict"]

        if v == "recyclable":
            st.markdown(
                '<div class="verdict-recyclable"><div class="v-icon">♻</div><div><div class="v-text">Recyclable</div><div class="v-sub">Place in the recycling bin</div></div></div>',
                unsafe_allow_html=True
            )

        elif v == "non-recyclable":
            st.markdown(
                '<div class="verdict-nonrecyclable"><div class="v-icon">✕</div><div><div class="v-text">Non-Recyclable</div><div class="v-sub">Place in the general waste bin</div></div></div>',
                unsafe_allow_html=True
            )

        elif v == "mixed":
            st.markdown(
                '<div class="verdict-mixed"><div class="v-icon">⚠</div><div><div class="v-text">Mixed Waste</div><div class="v-sub">Separate items before disposal</div></div></div>',
                unsafe_allow_html=True
            )

        else:
            st.warning("Nothing detected. Try using a clearer image with better lighting.")

        if r["detections"]:
            st.markdown("""
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;
                color:#404040;margin-top:1.5rem;margin-bottom:1rem;font-family:'Syne',sans-serif;">
                Item breakdown
            </div>""", unsafe_allow_html=True)

            RECYCLABLE_CLASSES, FRIENDLY = build_class_maps(model)

            for d in r["detections"]:
                is_rec = d["class"] in RECYCLABLE_CLASSES
                friendly = FRIENDLY.get(d["class"], d["class"].title())
                tip = DISPOSAL_TIPS.get(d["class"], DEFAULT_TIP)

                chip_cls = "chip-rec" if is_rec else "chip-nonrec"
                chip_lbl = "Recyclable" if is_rec else "Non-recyclable"

                fill_col = "#4ade80" if is_rec else "#f87171"
                pct = d["conf"] * 100

                st.markdown(
                    f'<div class="det-item">'
                    f'<div class="det-top"><span class="det-name">{friendly}<span class="rec-chip {chip_cls}">{chip_lbl}</span></span><span class="det-conf">{pct:.1f}%</span></div>'
                    f'<div class="conf-track"><div style="height:7px;border-radius:3px;background:{fill_col};width:{pct:.1f}%"></div></div>'
                    f'<div class="det-tip">{tip}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        st.download_button(
            "Download annotated image",
            data=r["annotated_bytes"],
            file_name="wastelens_result.png",
            mime="image/png",
            use_container_width=True
        )
