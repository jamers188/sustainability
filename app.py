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
    with st.spinner("Downloading model weights..."):
        download_model(GDRIVE_FILE_ID, MODEL_PATH)

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        st.error("Download failed — check GDRIVE_FILE_ID and that the file is shared publicly.")
        st.stop()

st.set_page_config(
    page_title="WasteLens",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800;900&family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header, [data-testid="stSidebar"] {
    visibility: hidden !important;
    display: none !important;
}

.stApp {
    background:
        radial-gradient(circle at 20% 5%, rgba(34,197,94,0.11), transparent 26rem),
        radial-gradient(circle at 80% 30%, rgba(34,197,94,0.045), transparent 24rem),
        radial-gradient(circle at 85% 85%, rgba(34,197,94,0.055), transparent 30rem),
        linear-gradient(135deg, #030504 0%, #07100c 50%, #020302 100%);
    color: #f9fafb;
}

.block-container {
    max-width: 1120px !important;
    padding: 1.7rem 2rem 3.2rem !important;
}

/* HERO */
.hero {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 1.2rem;
    margin-bottom: 2.7rem;
    padding-bottom: 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.065);
}

.logo {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 4vw, 4rem);
    font-weight: 900;
    letter-spacing: -0.08em;
    color: #ffffff;
    line-height: 0.9;
}

.logo span {
    color: #22c55e;
}

.subtitle {
    margin-top: 0.75rem;
    color: #9ca3af;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
}

.model-pill {
    color: #4ade80;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.5);
    border-radius: 999px;
    padding: 0.5rem 0.85rem;
    font-size: 0.66rem;
    font-weight: 900;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

/* SECTION WRAPS */
.input-card,
.result-card {
    background: rgba(255,255,255,0.018);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 22px;
    padding: 1.15rem;
    min-height: 330px;
}

.input-card {
    box-shadow: inset 0 0 40px rgba(34,197,94,0.018);
}

.result-card {
    box-shadow: inset 0 0 55px rgba(34,197,94,0.025);
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 900;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #e5e7eb;
    margin-bottom: 0.95rem;
}

/* RADIO */
[data-testid="stRadio"] > label {
    display: none !important;
}

[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    gap: 0.65rem !important;
}

[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    padding: 0.6rem 0.85rem !important;
}

[data-testid="stRadio"] label:hover {
    border-color: rgba(34,197,94,0.65) !important;
}

/* UPLOAD */
[data-testid="stFileUploadDropzone"] {
    border: 1.5px dashed rgba(34,197,94,0.75) !important;
    border-radius: 14px !important;
    background: rgba(15,15,15,0.96) !important;
    padding: 1rem !important;
    min-height: 95px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(34,197,94,0.06) !important;
    box-shadow: inset 0 0 30px rgba(34,197,94,0.04), 0 0 20px rgba(34,197,94,0.08) !important;
}

[data-testid="stFileUploadDropzone"] p {
    color: #d1d5db !important;
    font-size: 0.84rem !important;
    text-align: center !important;
}

[data-testid="stFileUploadDropzone"] small {
    color: #6b7280 !important;
    font-size: 0.72rem !important;
}

[data-testid="stFileUploadDropzone"] svg {
    color: #22c55e !important;
    fill: #22c55e !important;
    width: 1.8rem !important;
    height: 1.8rem !important;
}

/* IMAGES */
img {
    border-radius: 14px !important;
}

/* BUTTONS */
.stButton > button {
    background: #22c55e !important;
    color: #000 !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 900 !important;
    font-size: 0.82rem !important;
    height: auto !important;
    padding: 0.85rem 1rem !important;
    box-shadow: 0 14px 35px rgba(34,197,94,0.24) !important;
    opacity: 1 !important;
}

.stButton > button:not(:disabled) {
    background: #22c55e !important;
    color: #000 !important;
}

.stButton > button:hover {
    background: #4ade80 !important;
    color: #000 !important;
    transform: translateY(-1px) !important;
}

.stButton > button:disabled {
    background: #1a1a1a !important;
    color: #4b5563 !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

[data-testid="stDownloadButton"] > button {
    background: rgba(34,197,94,0.06) !important;
    color: #4ade80 !important;
    border: 1px solid rgba(34,197,94,0.45) !important;
    border-radius: 13px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 900 !important;
    width: 100% !important;
    height: auto !important;
    padding: 0.85rem 1rem !important;
}

/* AWAIT PANEL */
.await-panel {
    background:
        radial-gradient(circle at center, rgba(34,197,94,0.08), transparent 13rem),
        rgba(255,255,255,0.025);
    border: 1px dashed rgba(255,255,255,0.13);
    border-radius: 18px;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
}

.await-icon {
    font-size: 3rem;
    margin-bottom: 0.9rem;
    color: #22c55e;
    text-shadow: 0 0 24px rgba(34,197,94,0.4);
}

.await-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 900;
    letter-spacing: 0.17em;
    text-transform: uppercase;
    color: #f3f4f6;
    margin-bottom: 0.55rem;
}

.await-text {
    color: #9ca3af;
    font-size: 0.82rem;
    line-height: 1.65;
}

/* METRICS */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 0.7rem;
    margin: 0.9rem 0 1rem;
}

.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1rem 0.8rem;
    text-align: center;
}

.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 900;
    color: #fff;
    line-height: 1;
}

.metric-lbl {
    font-size: 0.55rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.45rem;
    font-weight: 800;
}

/* VERDICTS */
.verdict-recyclable,
.verdict-nonrecyclable,
.verdict-mixed {
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.verdict-recyclable {
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.04));
    border: 1px solid rgba(34,197,94,0.42);
}

.verdict-nonrecyclable {
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.04));
    border: 1px solid rgba(239,68,68,0.42);
}

.verdict-mixed {
    background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(245,158,11,0.04));
    border: 1px solid rgba(245,158,11,0.42);
}

.v-icon {
    font-size: 1.5rem;
}

.v-text {
    font-family: 'Syne', sans-serif;
    font-weight: 900;
    font-size: 0.88rem;
    color: #f9fafb;
}

.v-sub {
    font-size: 0.76rem;
    color: #9ca3af;
    margin-top: 0.12rem;
}

/* BREAKDOWN */
.det-item {
    background: #111111;
    border: 1px solid #222222;
    border-radius: 13px;
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.6rem;
}

.det-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.55rem;
}

.det-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 800;
    color: #f3f4f6;
}

.det-conf {
    font-size: 0.72rem;
    color: #d1d5db;
    font-weight: 700;
}

.conf-track {
    background: #27272a;
    border-radius: 999px;
    height: 7px;
    overflow: hidden;
    margin-bottom: 0.6rem;
}

.conf-track div {
    height: 7px !important;
    border-radius: 999px !important;
}

.det-tip {
    color: #9ca3af;
    font-size: 0.72rem;
    line-height: 1.5;
}

.chip-rec,
.chip-nonrec {
    display: inline-block;
    font-size: 0.5rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    padding: 0.22rem 0.45rem;
    border-radius: 999px;
    margin-left: 0.45rem;
    vertical-align: middle;
}

.chip-rec {
    background: #166534;
    color: #4ade80;
    border: 1px solid #15803d;
}

.chip-nonrec {
    background: #7f1d1d;
    color: #fca5a5;
    border: 1px solid #991b1b;
}

.footer-line {
    text-align: center;
    color: #4b5563;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    margin-top: 2.2rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}

@media (max-width: 900px) {
    .hero {
        flex-direction: column;
        align-items: flex-start;
    }

    .metric-row {
        grid-template-columns: repeat(2, 1fr);
    }

    .input-card,
    .result-card {
        min-height: auto;
    }
}
</style>
""", unsafe_allow_html=True)

CONF_THRESH = 0.25
IOU_THRESH = 0.45
SHOW_LABELS = True
SHOW_CONF_IMG = True

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

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_image_key" not in st.session_state:
    st.session_state.last_image_key = None

st.markdown("""
<div class="hero">
    <div>
        <div class="logo">waste<span>lens</span></div>
        <div class="subtitle">AI-powered waste classification for smarter recycling decisions.</div>
    </div>
    <div class="model-pill">v4 model</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([0.92, 1.28], gap="large")

with left:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Input Source</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Input",
        ["Upload image", "Use camera"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    source_image = None
    current_image_key = None

    if mode == "Upload image":
        uploaded = st.file_uploader(
            "upload",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed"
        )

        if uploaded:
            current_image_key = f"upload_{uploaded.name}_{uploaded.size}"

            if st.session_state.last_image_key != current_image_key:
                st.session_state.last_result = None
                st.session_state.last_image_key = current_image_key

            tmp_path = f"/tmp/wastelens_upload_{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getvalue())

            source_image = Image.open(tmp_path).convert("RGB")
            st.session_state["img_path"] = tmp_path

    else:
        st.caption("Allow camera access in your browser, then click the shutter button.")
        cam = st.camera_input("camera", label_visibility="collapsed")

        if cam:
            current_image_key = f"camera_{len(cam.getvalue())}"

            if st.session_state.last_image_key != current_image_key:
                st.session_state.last_result = None
                st.session_state.last_image_key = current_image_key

            tmp_path = "/tmp/wastelens_camera.jpg"
            with open(tmp_path, "wb") as f:
                f.write(cam.getvalue())

            source_image = Image.open(tmp_path).convert("RGB")
            st.session_state["img_path"] = tmp_path

    if source_image:
        st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Original Image</div>', unsafe_allow_html=True)
        st.image(source_image, caption="Original image", use_container_width=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    run = st.button(
        "✦ Analyse Image",
        disabled=(source_image is None),
        use_container_width=True,
        type="primary"
    )

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if source_image is None:
        st.markdown(
            '<div class="await-panel">'
            '<div class="await-icon">♻</div>'
            '<div class="await-title">Awaiting Scan</div>'
            '<div class="await-text">Upload or capture an image on the left,<br>then press Analyse Image.</div>'
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

    r = st.session_state.last_result

    if source_image is not None and r is None:
        st.markdown(
            '<div class="await-panel">'
            '<div class="await-icon">♻</div>'
            '<div class="await-title">Ready To Analyse</div>'
            '<div class="await-text">Image loaded successfully.<br>Press Analyse Image to run YOLO detection.</div>'
            '</div>',
            unsafe_allow_html=True
        )

    if r and source_image is not None:
        st.markdown('<div class="section-title">Detection Overlay</div>', unsafe_allow_html=True)
        st.image(r["annotated_pil"], caption="Detection overlay", use_container_width=True)

        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-box"><div class="metric-val">{r["n_det"]}</div><div class="metric-lbl">Detected</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#4ade80">{r["rec_count"]}</div><div class="metric-lbl">Recyclable</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#f87171">{r["nonrec_count"]}</div><div class="metric-lbl">Non-Recyclable</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#e5e7eb">{r["elapsed"]*1000:.0f}ms</div><div class="metric-lbl">Inference</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        v = r["verdict"]

        if v == "recyclable":
            st.markdown(
                '<div class="verdict-recyclable"><div class="v-icon">♻</div><div><div class="v-text">Verdict: Recyclable</div><div class="v-sub">Place in the recycling bin.</div></div></div>',
                unsafe_allow_html=True
            )

        elif v == "non-recyclable":
            st.markdown(
                '<div class="verdict-nonrecyclable"><div class="v-icon">🗑</div><div><div class="v-text">Verdict: Non-Recyclable</div><div class="v-sub">Place in the general waste bin.</div></div></div>',
                unsafe_allow_html=True
            )

        elif v == "mixed":
            st.markdown(
                '<div class="verdict-mixed"><div class="v-icon">⚠</div><div><div class="v-text">Verdict: Mixed Waste</div><div class="v-sub">Separate items before disposal.</div></div></div>',
                unsafe_allow_html=True
            )

        else:
            st.warning("Nothing detected. Try using a clearer image with better lighting.")

        if r["detections"]:
            st.markdown('<div class="section-title" style="margin-top:1rem;">Item Breakdown</div>', unsafe_allow_html=True)

            RECYCLABLE_CLASSES, FRIENDLY = build_class_maps(model)

            for d in r["detections"]:
                is_rec = d["class"] in RECYCLABLE_CLASSES
                friendly = FRIENDLY.get(d["class"], d["class"].title())
                tip = DISPOSAL_TIPS.get(d["class"], DEFAULT_TIP)

                chip_cls = "chip-rec" if is_rec else "chip-nonrec"
                chip_lbl = "Recyclable" if is_rec else "Non-Recyclable"

                fill_col = "#4ade80" if is_rec else "#f87171"
                pct = d["conf"] * 100

                st.markdown(
                    f'<div class="det-item">'
                    f'<div class="det-top">'
                    f'<span class="det-name">{friendly}<span class="{chip_cls}">{chip_lbl}</span></span>'
                    f'<span class="det-conf">{pct:.1f}%</span>'
                    f'</div>'
                    f'<div class="conf-track"><div style="background:{fill_col};width:{pct:.1f}%"></div></div>'
                    f'<div class="det-tip">{tip}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)

        st.download_button(
            "⬇ Download Annotated Image",
            data=r["annotated_bytes"],
            file_name="wastelens_result.png",
            mime="image/png",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-line">WasteLens · AI Waste Classification</div>',
    unsafe_allow_html=True
)
