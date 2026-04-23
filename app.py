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
#  MODEL DOWNLOAD (for Streamlit Cloud)
#  The .pt file is too large for Git LFS
#  on Streamlit Cloud, so we pull it from
#  Google Drive at startup if not present.
# ─────────────────────────────────────────
MODEL_PATH = "waste_final_v4_best.pt"
# Paste ONLY the file ID from your Google Drive share link.
# Share link looks like: https://drive.google.com/file/d/1ABC123xyz.../view
# Copy ONLY the bold part:                                  ^^^^^^^^^^^
GDRIVE_FILE_ID = "1oV2jz3IDv1_8M_DNySn3dsMj3KyPyl2c"

def download_model(file_id: str, dest: str):
    import requests
    session = requests.Session()
    # Step 1: hit the export URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = session.get(url, stream=True)
    # Step 2: Google redirects large files through a confirm page
    # The confirm token is now in the response URL or a cookie
    token = None
    for k, v in r.cookies.items():
        if "download_warning" in k:
            token = v
    if token:
        r = session.get(url + f"&confirm={token}", stream=True)
    # Step 3: also handle the newer "confirm=t" style redirect
    if "text/html" in r.headers.get("Content-Type", ""):
        # parse confirm from HTML
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
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="WasteLens",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── RESET ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 4rem !important; max-width: 1400px !important; }
section[data-testid="stSidebar"] > div { padding-top: 2rem !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #050505 !important;
    border-right: 1px solid #111 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: #555 !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #1a1a1a !important; }

/* ── UPLOAD DROPZONE ── */
[data-testid="stFileUploadDropzone"] {
    background: #080808 !important;
    border: 1.5px dashed #1e1e1e !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #22c55e !important;
    background: #0a120c !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span { color: #333 !important; font-size: 0.8rem !important; }
[data-testid="stFileUploadDropzone"] svg { stroke: #222 !important; }

/* ── BUTTONS ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    height: 3rem !important;
    box-shadow: 0 4px 24px rgba(34,197,94,0.25) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(34,197,94,0.4) !important;
}
.stButton > button:disabled {
    background: #111 !important;
    color: #222 !important;
    box-shadow: none !important;
    transform: none !important;
}
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #22c55e !important;
    border: 1px solid #1a3a24 !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    width: 100% !important;
    height: 2.6rem !important;
    transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #0a1f10 !important;
    border-color: #22c55e !important;
}

/* ── RADIO ── */
[data-testid="stRadio"] > label { display: none !important; }
[data-testid="stRadio"] div[role="radiogroup"] {
    background: #080808;
    border: 1px solid #111;
    border-radius: 10px;
    padding: 4px;
    display: flex;
    gap: 4px;
}
[data-testid="stRadio"] label {
    flex: 1;
    text-align: center;
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
    cursor: pointer;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #444 !important;
    transition: all 0.15s !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #0f2d1a !important;
    color: #22c55e !important;
}

/* ── CAMERA ── */
[data-testid="stCameraInput"] { border-radius: 16px !important; overflow: hidden; }

/* ── IMAGES ── */
[data-testid="stImage"] img { border-radius: 12px !important; }

/* ── SPINNERS / ALERTS ── */
[data-testid="stAlert"] {
    background: #0a0a0a !important;
    border-color: #1a1a1a !important;
    border-radius: 10px !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #050505; }
::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────
# These are matched against model.names dynamically — see build_class_maps() below
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
    st.markdown("""<div style="font-size:0.58rem;font-weight:700;letter-spacing:0.2em;
        text-transform:uppercase;color:#22c55e;border-bottom:1px solid #1a1a1a;
        padding-bottom:0.5rem;margin-bottom:0.8rem;font-family:'Outfit',sans-serif;">
        Settings</div>""", unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence threshold", 0.01, 0.50, 0.25, 0.01,
                            help="Lower = more detections, more false positives")
    iou_thresh  = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05,
                            help="Controls overlap suppression")
    show_labels  = st.checkbox("Show labels on image", value=True)
    show_conf_img= st.checkbox("Show confidence on image", value=True)

    st.markdown("""<div style="font-size:0.58rem;font-weight:700;letter-spacing:0.2em;
        text-transform:uppercase;color:#22c55e;border-bottom:1px solid #1a1a1a;
        padding-bottom:0.5rem;margin-bottom:0.8rem;margin-top:1.5rem;font-family:'Outfit',sans-serif;">
        Detection History</div>""", unsafe_allow_html=True)
    if st.session_state.history:
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
        cmap = {"recyclable":"#7ed4a0","non-recyclable":"#e07070","mixed":"#d4b86a","no-detection":"#6b7280"}
        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            num = len(st.session_state.history) - i
            c = cmap.get(h["verdict"], "#6b7280")
            st.markdown(
                f'<div class="hist-item">'
                f'<span style="color:#4b5563;font-family:Space Mono,monospace;font-size:0.62rem">#{num}</span>'
                f'<span style="color:#9ca3af">{h["count"]} obj</span>'
                f'<span style="color:{c};font-family:Space Mono,monospace;font-size:0.65rem">{h["verdict"].upper()}</span>'
                f'</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#4b5563;font-size:0.78rem">No scans yet.</p>', unsafe_allow_html=True)



# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="padding:3rem 0 2.5rem;border-bottom:1px solid #0f0f0f;margin-bottom:2.5rem;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
        <div style="display:flex;align-items:baseline;gap:0.15em;">
            <span style="font-family:'Outfit',sans-serif;font-size:3.2rem;font-weight:800;
                color:#fff;letter-spacing:-0.04em;line-height:1;">Waste</span>
            <span style="font-family:'Outfit',sans-serif;font-size:3.2rem;font-weight:800;
                color:#22c55e;letter-spacing:-0.04em;line-height:1;">Lens</span>
            <span style="display:inline-flex;align-items:center;margin-left:1rem;
                font-size:0.55rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;
                color:#22c55e;background:#071a0d;border:1px solid #14532d;
                padding:5px 12px;border-radius:100px;vertical-align:middle;
                position:relative;top:-0.5rem;">AI</span>
        </div>
    </div>
    <div style="font-size:0.72rem;color:#2a2a2a;letter-spacing:0.3em;
        text-transform:uppercase;font-weight:500;">
        Point &nbsp;/&nbsp; Detect &nbsp;/&nbsp; Sort correctly
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.22em;text-transform:uppercase;
        color:#1e1e1e;margin-bottom:0.8rem;font-family:'Outfit',sans-serif;">
        Input source
    </div>""", unsafe_allow_html=True)
    mode = st.radio("Input", ["Upload image", "Use camera"], horizontal=True, label_visibility="collapsed")
    source_image = None

    if mode == "Upload image":
        uploaded = st.file_uploader("upload", type=["jpg","jpeg","png","webp","bmp"], label_visibility="collapsed")
        if uploaded:
            # Save raw bytes to a temp file — avoids PIL conversion artifacts
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
        st.image(source_image, caption="Original image", use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Analyse image", disabled=(source_image is None), use_container_width=True, type="primary")

with right:
    if source_image is None:
        st.markdown("""
        <div style="
            background:#050505;border:1px solid #0f0f0f;border-radius:20px;
            min-height:360px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;text-align:center;padding:3rem;
        ">
            <div style="width:56px;height:56px;border-radius:16px;background:#0a0a0a;
                border:1px solid #141414;display:flex;align-items:center;justify-content:center;
                font-size:1.5rem;margin-bottom:1.5rem;">&#9843;</div>
            <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.25em;
                text-transform:uppercase;color:#1a1a1a;margin-bottom:0.6rem;
                font-family:'Outfit',sans-serif;">Ready to scan</div>
            <div style="font-size:0.82rem;color:#1f1f1f;line-height:1.7;max-width:220px;">
                Upload an image or use your camera, then press Analyse
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif run:
        with st.spinner("Running detection..."):
            t0 = time.time()
            # Pass the file path directly — identical to how Colab runs it
            # This bypasses any PIL/numpy conversion quality loss
            img_path = st.session_state.get("img_path")
            predict_input = img_path if img_path and os.path.exists(img_path) else np.array(source_image)
            results = model.predict(predict_input, conf=conf_thresh, iou=iou_thresh, imgsz=640, verbose=False)
            elapsed = time.time() - t0



        boxes = results[0].boxes
        names = results[0].names
        n_det = len(boxes)
        detections = [{"class": names[int(b.cls[0])].lower(), "conf": float(b.conf[0])} for b in boxes]

        # Build class maps dynamically from this model's actual names
        RECYCLABLE_CLASSES, FRIENDLY = build_class_maps(model)

        annotated_pil = Image.fromarray(results[0].plot(labels=show_labels, conf=show_conf_img))
        rec    = [d for d in detections if d["class"] in RECYCLABLE_CLASSES]
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
            "annotated_bytes": buf.getvalue(), "annotated_pil": annotated_pil,
            "detections": detections, "verdict": verdict, "elapsed": elapsed,
            "n_det": n_det, "rec_count": len(rec), "nonrec_count": len(nonrec),
        }
        st.session_state.history.append({"count": n_det, "verdict": verdict})

    r = st.session_state.last_result
    if r and source_image is not None:
        st.image(r["annotated_pil"], caption="Detection overlay", use_column_width=True)

        st.markdown(f'''
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;margin-bottom:1.5rem;">
            <div style="background:#050505;border:1px solid #0f0f0f;border-radius:14px;
                padding:1.1rem 0.8rem;text-align:center;">
                <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;
                    color:#fff;line-height:1;">{r["n_det"]}</div>
                <div style="font-size:0.58rem;color:#222;text-transform:uppercase;
                    letter-spacing:0.15em;margin-top:6px;font-weight:600;">Detected</div>
            </div>
            <div style="background:#071a0d;border:1px solid #0f2d1a;border-radius:14px;
                padding:1.1rem 0.8rem;text-align:center;">
                <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;
                    color:#22c55e;line-height:1;">{r["rec_count"]}</div>
                <div style="font-size:0.58rem;color:#14532d;text-transform:uppercase;
                    letter-spacing:0.15em;margin-top:6px;font-weight:600;">Recyclable</div>
            </div>
            <div style="background:#1a0505;border:1px solid #2a0a0a;border-radius:14px;
                padding:1.1rem 0.8rem;text-align:center;">
                <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;
                    color:#ef4444;line-height:1;">{r["nonrec_count"]}</div>
                <div style="font-size:0.58rem;color:#7f1d1d;text-transform:uppercase;
                    letter-spacing:0.15em;margin-top:6px;font-weight:600;">Non-recyclable</div>
            </div>
            <div style="background:#050505;border:1px solid #0f0f0f;border-radius:14px;
                padding:1.1rem 0.8rem;text-align:center;">
                <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;
                    color:#333;line-height:1;">{r["elapsed"]*1000:.0f}<span style="font-size:0.9rem;">ms</span></div>
                <div style="font-size:0.58rem;color:#1a1a1a;text-transform:uppercase;
                    letter-spacing:0.15em;margin-top:6px;font-weight:600;">Inference</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        v = r["verdict"]
        if v == "recyclable":
            st.markdown('''
            <div style="background:linear-gradient(135deg,#071a0d,#0d2818);
                border:1px solid #14532d;border-radius:14px;padding:1.2rem 1.4rem;
                margin-bottom:1rem;display:flex;align-items:center;gap:1rem;">
                <div style="width:40px;height:40px;background:#0f2d1a;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;">♻</div>
                <div>
                    <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.95rem;
                        color:#22c55e;letter-spacing:-0.01em;">Recyclable</div>
                    <div style="font-size:0.72rem;color:#166534;margin-top:2px;">Place in the recycling bin</div>
                </div>
            </div>''', unsafe_allow_html=True)
        elif v == "non-recyclable":
            st.markdown('''
            <div style="background:linear-gradient(135deg,#1a0505,#200a0a);
                border:1px solid #7f1d1d;border-radius:14px;padding:1.2rem 1.4rem;
                margin-bottom:1rem;display:flex;align-items:center;gap:1rem;">
                <div style="width:40px;height:40px;background:#2a0a0a;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;">✕</div>
                <div>
                    <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.95rem;
                        color:#ef4444;letter-spacing:-0.01em;">Non-Recyclable</div>
                    <div style="font-size:0.72rem;color:#b91c1c;margin-top:2px;">Place in the general waste bin</div>
                </div>
            </div>''', unsafe_allow_html=True)
        elif v == "mixed":
            st.markdown('''
            <div style="background:linear-gradient(135deg,#1a1200,#221900);
                border:1px solid #92400e;border-radius:14px;padding:1.2rem 1.4rem;
                margin-bottom:1rem;display:flex;align-items:center;gap:1rem;">
                <div style="width:40px;height:40px;background:#2a1f00;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;">⚠</div>
                <div>
                    <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.95rem;
                        color:#f59e0b;letter-spacing:-0.01em;">Mixed Waste</div>
                    <div style="font-size:0.72rem;color:#b45309;margin-top:2px;">Separate items before disposal</div>
                </div>
            </div>''', unsafe_allow_html=True)
        else:
            st.warning("Nothing detected. Try lowering the confidence threshold in the sidebar.")

        if r["detections"]:
            st.markdown("""
            <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.22em;text-transform:uppercase;
                color:#1e1e1e;margin-top:1.5rem;margin-bottom:1rem;font-family:'Outfit',sans-serif;">
                Item breakdown
            </div>""", unsafe_allow_html=True)
            RECYCLABLE_CLASSES, FRIENDLY = build_class_maps(model)
            for d in r["detections"]:
                is_rec   = d["class"] in RECYCLABLE_CLASSES
                friendly = FRIENDLY.get(d["class"], d["class"].title())
                tip      = DISPOSAL_TIPS.get(d["class"], DEFAULT_TIP)
                chip_cls = "chip-rec" if is_rec else "chip-nonrec"
                chip_lbl = "Recyclable" if is_rec else "Non-recyclable"
                fill_col = "#5a9e6f" if is_rec else "#c05050"
                pct      = d["conf"] * 100
                st.markdown(f'''
                <div style="background:#050505;border:1px solid #0f0f0f;border-radius:14px;
                    padding:1rem 1.2rem;margin-bottom:0.6rem;">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                        <div style="display:flex;align-items:center;gap:0.6rem;">
                            <span style="font-family:'Outfit',sans-serif;font-size:0.88rem;
                                font-weight:700;color:#e5e5e5;">{friendly}</span>
                            <span style="font-size:0.58rem;font-weight:700;letter-spacing:0.1em;
                                padding:2px 8px;border-radius:100px;
                                background:{"#071a0d" if is_rec else "#1a0505"};
                                color:{"#22c55e" if is_rec else "#ef4444"};
                                border:1px solid {"#14532d" if is_rec else "#7f1d1d"};">
                                {"Recyclable" if is_rec else "Non-recyclable"}</span>
                        </div>
                        <span style="font-family:'Outfit',sans-serif;font-size:0.88rem;
                            font-weight:700;color:{"#22c55e" if is_rec else "#ef4444"};">{pct:.0f}%</span>
                    </div>
                    <div style="background:#0a0a0a;border-radius:4px;height:3px;margin-bottom:10px;">
                        <div style="height:3px;border-radius:4px;
                            background:{"linear-gradient(90deg,#16a34a,#22c55e)" if is_rec else "linear-gradient(90deg,#dc2626,#ef4444)"};
                            width:{pct:.0f}%;"></div>
                    </div>
                    <div style="font-size:0.72rem;color:#252525;line-height:1.6;">{tip}</div>
                </div>
                ''', unsafe_allow_html=True)

        st.download_button("Download annotated image", data=r["annotated_bytes"],
                           file_name="wastelens_result.png", mime="image/png", use_container_width=True)
