import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import io
import os
import requests

# ─────────────────────────────────────────
#  MODEL DOWNLOAD (for Streamlit Cloud)
#  The .pt file is too large for Git LFS
#  on Streamlit Cloud, so we pull it from
#  Google Drive at startup if not present.
# ─────────────────────────────────────────
MODEL_PATH = "waste_final_best.pt"
# Paste ONLY the file ID from your Google Drive share link.
# Share link looks like: https://drive.google.com/file/d/1ABC123xyz.../view
# Copy ONLY the bold part:                                  ^^^^^^^^^^^
GDRIVE_FILE_ID = "1cPShIOc70HPUEIb06fN4q0CcN9Ffw5n4"

def download_model(file_id: str, dest: str):
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)
    import gdown
    gdown.download(id=file_id, output=dest, quiet=False, fuzzy=True)

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
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"], p, div, span, label {
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.8rem 2rem 4rem !important; max-width: 1280px !important; }

.wl-wordmark {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.75rem; font-weight: 700; color: #ffffff;
    letter-spacing: -0.02em; margin: 0; line-height: 1;
}
.wl-tagline { font-size: 0.82rem; color: #6b7280; margin-top: 0.25rem; font-weight: 300; }
.wl-badge {
    display: inline-block; font-family: 'Space Mono', monospace;
    font-size: 0.58rem; letter-spacing: 0.18em; text-transform: uppercase;
    background: #1a2e22; color: #5a9e6f; border: 1px solid #2d4a38;
    padding: 3px 9px; border-radius: 2px; vertical-align: middle; margin-left: 0.6rem;
}
.wl-divider { border: none; border-top: 1px solid #252830; margin: 1rem 0 1.6rem 0; }
.wl-section {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: #6b7280;
    margin-bottom: 0.75rem; margin-top: 0.25rem;
}

.verdict-recyclable {
    background: #1a2e22; border: 1px solid #2d4a38; border-left: 3px solid #5a9e6f;
    border-radius: 4px; padding: 0.9rem 1.1rem; color: #7ed4a0;
    font-family: 'Space Mono', monospace; font-size: 0.8rem; margin-bottom: 1rem;
}
.verdict-nonrecyclable {
    background: #2a1a1a; border: 1px solid #4a2d2d; border-left: 3px solid #c05050;
    border-radius: 4px; padding: 0.9rem 1.1rem; color: #e07070;
    font-family: 'Space Mono', monospace; font-size: 0.8rem; margin-bottom: 1rem;
}
.verdict-mixed {
    background: #1e1e10; border: 1px solid #3a3520; border-left: 3px solid #c0a040;
    border-radius: 4px; padding: 0.9rem 1.1rem; color: #d4b86a;
    font-family: 'Space Mono', monospace; font-size: 0.8rem; margin-bottom: 1rem;
}

.metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.6rem; margin-bottom: 1.2rem; }
.metric-box { background: #0e0f11; border: 1px solid #252830; border-radius: 4px; padding: 0.9rem 0.7rem; text-align: center; }
.metric-val { font-family: 'Space Mono', monospace; font-size: 1.45rem; font-weight: 700; color: #ffffff; line-height: 1; }
.metric-lbl { font-size: 0.65rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

.det-item { background: #0e0f11; border: 1px solid #252830; border-radius: 4px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
.det-name { font-family: 'Space Mono', monospace; font-size: 0.78rem; color: #e8e6e1; margin-bottom: 2px; }
.det-meta { font-size: 0.72rem; color: #6b7280; margin-bottom: 6px; }
.conf-track { background: #252830; border-radius: 2px; height: 4px; }
.det-tip { font-size: 0.72rem; color: #9ca3af; margin-top: 6px; border-top: 1px solid #1c1e24; padding-top: 5px; }
.rec-chip { display: inline-block; font-size: 0.6rem; font-family: 'Space Mono', monospace; letter-spacing: 0.1em; padding: 1px 6px; border-radius: 2px; margin-left: 6px; vertical-align: middle; }
.chip-rec    { background:#1a2e22; color:#7ed4a0; border:1px solid #2d4a38; }
.chip-nonrec { background:#2a1a1a; color:#e07070; border:1px solid #4a2d2d; }

.await-panel {
    background: #13151a; border: 1px solid #252830; border-radius: 6px;
    min-height: 280px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center; padding: 2rem;
}
.await-bracket { font-family: 'Space Mono', monospace; font-size: 3rem; color: #252830; line-height: 1; margin-bottom: 0.8rem; }
.await-text { font-size: 0.82rem; color: #4b5563; }

.hist-item { display: flex; justify-content: space-between; align-items: center; padding: 0.45rem 0; border-bottom: 1px solid #1c1e24; font-size: 0.78rem; }
.hist-item:last-child { border-bottom: none; }
.sb-label {
    font-family: 'Space Mono', monospace; font-size: 0.6rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: #5a9e6f; border-bottom: 1px solid #252830;
    padding-bottom: 0.4rem; margin-bottom: 0.75rem; margin-top: 0.5rem;
}
.tip-box { background: #0e1218; border: 1px dashed #252830; border-radius: 4px; padding: 0.7rem 0.9rem; font-size: 0.75rem; color: #6b7280; line-height: 1.7; }

[data-testid="stRadio"] > label { display: none !important; }
.stButton > button { font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; border-radius: 3px !important; }
[data-testid="stDownloadButton"] > button { font-family: 'Space Mono', monospace !important; font-size: 0.68rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; border-radius: 3px !important; width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────
# These are matched against model.names dynamically — see build_class_maps() below
RECYCLABLE_KEYWORDS = ["can", "glass", "paper", "plastic", "cardboard", "bottle", "metal"]

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
        if any(kw in n for kw in RECYCLABLE_KEYWORDS):
            recyclable.add(n)
    return recyclable, friendly

# ─────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("waste_final_best.pt")

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
    st.markdown('<div class="sb-label">Settings</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05,
                            help="Lower = more detections, more false positives")
    iou_thresh  = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05,
                            help="Controls overlap suppression")
    show_labels  = st.checkbox("Show labels on image", value=True)
    show_conf_img= st.checkbox("Show confidence on image", value=True)

    st.markdown('<div class="sb-label" style="margin-top:1.5rem">Detection History</div>', unsafe_allow_html=True)
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

    st.markdown('<br>', unsafe_allow_html=True)
    with st.expander("Model debug info"):
        st.markdown('<div class="wl-section">Actual model class names</div>', unsafe_allow_html=True)
        for idx, name in model.names.items():
            st.markdown(f'`{idx}` → `{name}`')
        st.caption("If nothing detects, check these names match your waste objects.")


# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown(
    '<p class="wl-wordmark">WasteLens<span class="wl-badge">AI</span></p>'
    '<p class="wl-tagline">Point, detect, sort correctly.</p>'
    '<hr class="wl-divider">',
    unsafe_allow_html=True)

# ─────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="wl-section">Input source</div>', unsafe_allow_html=True)
    mode = st.radio("Input", ["Upload image", "Use camera"], horizontal=True, label_visibility="collapsed")
    source_image = None

    if mode == "Upload image":
        uploaded = st.file_uploader("upload", type=["jpg","jpeg","png","webp","bmp"], label_visibility="collapsed")
        if uploaded:
            source_image = Image.open(uploaded).convert("RGB")
    else:
        st.caption("Allow camera access in your browser, then click the shutter button.")
        cam = st.camera_input("camera", label_visibility="collapsed")
        if cam:
            source_image = Image.open(cam).convert("RGB")

    if source_image:
        st.image(source_image, caption="Original image", use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Analyse image", disabled=(source_image is None), use_container_width=True, type="primary")

with right:
    if source_image is None:
        st.markdown(
            '<div class="await-panel">'
            '<div class="await-bracket">[ ]</div>'
            '<div class="wl-section" style="margin-bottom:0.4rem">Awaiting scan</div>'
            '<div class="await-text">Upload or capture an image on the left,<br>then press Analyse.</div>'
            '</div>', unsafe_allow_html=True)

    elif run:
        with st.spinner("Running detection..."):
            t0 = time.time()
            img_array = np.array(source_image)
            results = model.predict(img_array, conf=conf_thresh, iou=iou_thresh, imgsz=640, verbose=False)
            elapsed = time.time() - t0

        # ── DEBUG PANEL — shows exactly what's happening ──────────────
        with st.expander("Debug info (expand to diagnose)", expanded=True):
            model_size = os.path.getsize(MODEL_PATH) / 1e6
            st.write(f"**Model file:** `{MODEL_PATH}` — `{model_size:.2f} MB` {'OK' if model_size > 1 else 'PROBLEM: file too small!'}")
            st.write(f"**Model classes:** `{model.names}`")
            st.write(f"**Image:** size={source_image.size} mode={source_image.mode}")
            st.write(f"**Detections at your conf={conf_thresh}:** `{len(results[0].boxes)}`")

            st.write("**Scanning at conf=0.01 (lowest) across image sizes:**")
            for sz in [640, 1280, 416]:
                r2 = model.predict(np.array(source_image), conf=0.01, iou=0.45, imgsz=sz, verbose=False)
                hits = [(model.names[int(b.cls[0])], round(float(b.conf[0]),3)) for b in r2[0].boxes]
                st.write(f"  imgsz={sz} → {len(r2[0].boxes)} detection(s): {hits}")
        # ── END DEBUG ─────────────────────────────────────────────────

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

        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-box"><div class="metric-val">{r["n_det"]}</div><div class="metric-lbl">Detected</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#7ed4a0">{r["rec_count"]}</div><div class="metric-lbl">Recyclable</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#e07070">{r["nonrec_count"]}</div><div class="metric-lbl">Non-recyclable</div></div>'
            f'<div class="metric-box"><div class="metric-val" style="color:#9ca3af">{r["elapsed"]*1000:.0f}ms</div><div class="metric-lbl">Inference</div></div>'
            f'</div>', unsafe_allow_html=True)

        v = r["verdict"]
        if v == "recyclable":
            st.markdown('<div class="verdict-recyclable">RECYCLABLE — Place in the recycling bin.</div>', unsafe_allow_html=True)
        elif v == "non-recyclable":
            st.markdown('<div class="verdict-nonrecyclable">NON-RECYCLABLE — General waste bin.</div>', unsafe_allow_html=True)
        elif v == "mixed":
            st.markdown('<div class="verdict-mixed">MIXED WASTE — Separate items before disposal.</div>', unsafe_allow_html=True)
        else:
            st.warning("Nothing detected. Try lowering the confidence threshold in the sidebar.")

        if r["detections"]:
            st.markdown('<div class="wl-section" style="margin-top:0.8rem">Item breakdown</div>', unsafe_allow_html=True)
            RECYCLABLE_CLASSES, FRIENDLY = build_class_maps(model)
            for d in r["detections"]:
                is_rec   = d["class"] in RECYCLABLE_CLASSES
                friendly = FRIENDLY.get(d["class"], d["class"].title())
                tip      = DISPOSAL_TIPS.get(d["class"], DEFAULT_TIP)
                chip_cls = "chip-rec" if is_rec else "chip-nonrec"
                chip_lbl = "Recyclable" if is_rec else "Non-recyclable"
                fill_col = "#5a9e6f" if is_rec else "#c05050"
                pct      = d["conf"] * 100
                st.markdown(
                    f'<div class="det-item">'
                    f'<div class="det-name">{friendly}<span class="rec-chip {chip_cls}">{chip_lbl}</span></div>'
                    f'<div class="det-meta">Confidence: {pct:.1f}%</div>'
                    f'<div class="conf-track"><div style="height:4px;border-radius:2px;background:{fill_col};width:{pct:.1f}%"></div></div>'
                    f'<div class="det-tip">{tip}</div>'
                    f'</div>', unsafe_allow_html=True)

        st.download_button("Download annotated image", data=r["annotated_bytes"],
                           file_name="wastelens_result.png", mime="image/png", use_container_width=True)
