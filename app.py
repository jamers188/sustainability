import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["MPLBACKEND"] = "Agg"

import streamlit as st
from PIL import Image
import numpy as np
import time
import io
import requests
import re
from ultralytics import YOLO

# ── MODEL DOWNLOAD ──────────────────────────────────────────────
MODEL_PATH = "best_model.pt"
GDRIVE_FILE_ID = "1FYO7H9UnLDuw5FwAqVpLSvEnPC1dTmod"

def download_model(file_id, dest):
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
        match = re.search(r'confirm=([0-9A-Za-z_\-]+)', r.text)
        if match:
            r = session.get(url + f"&confirm={match.group(1)}", stream=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

model_ok = os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000
if not model_ok:
    st.set_page_config(page_title="WasteLens", layout="wide")
    with st.spinner("Downloading model weights..."):
        download_model(GDRIVE_FILE_ID, MODEL_PATH)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        st.error("Model download failed.")
        st.stop()
    st.rerun()

# ── PAGE CONFIG ─────────────────────────────────────────────────
st.set_page_config(
    page_title="WasteLens — AI Waste Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CONSTANTS ───────────────────────────────────────────────────
RECYCLABLE_KEYWORDS = ["can", "glass", "paper", "plastic", "cardboard", "bottle", "metal"]
NON_RECYCLABLE_CLASSES = {"foodwaste", "food", "organic", "food_waste"}

DISPOSAL_TIPS = {
    "cans": "Rinse before recycling. Crush to save bin space.",
    "glass": "Remove lids. Sort by color if required.",
    "paperwaste": "Keep dry. Remove staples and plastic film.",
    "plasticbottles": "Empty, rinse, check resin code on base.",
    "foodwaste": "Seal in compostable bag. Use organic bin.",
}
DEFAULT_TIP = "Seal and place in the general waste bin."

CLASS_COLORS = {
    "plasticbottles": "#3b82f6",
    "glass": "#8b5cf6",
    "paperwaste": "#f59e0b",
    "cans": "#22c55e",
    "foodwaste": "#ef4444",
}

def build_class_maps(model):
    recyclable, friendly = set(), {}
    for idx, name in model.names.items():
        n = name.lower()
        friendly[n] = name.replace("_", " ").replace("waste", "").strip().title()
        if any(kw in n for kw in RECYCLABLE_KEYWORDS) and n not in NON_RECYCLABLE_CLASSES:
            recyclable.add(n)
    return recyclable, friendly

# ── MODEL ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ── SESSION STATE ────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── GLOBAL CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; background: #030303 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080808 !important;
    border-right: 1px solid #111 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* Hide default radio label */
[data-testid="stRadio"] > label { display: none !important; }

/* Upload dropzone */
[data-testid="stFileUploadDropzone"] {
    background: #0d0d0d !important;
    border: 1.5px dashed #1a1a1a !important;
    border-radius: 14px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: #22c55e !important; background: #0a120c !important; }
[data-testid="stFileUploadDropzone"] p { color: #2a2a2a !important; font-size: 0.82rem !important; }
[data-testid="stFileUploadDropzone"] svg { stroke: #1a1a1a !important; }
[data-testid="stFileUploadDropzone"] small { color: #1a1a1a !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    color: #000 !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important; font-size: 0.82rem !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    height: 3rem !important; width: 100% !important;
    box-shadow: 0 0 24px rgba(34,197,94,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { box-shadow: 0 0 36px rgba(34,197,94,0.5) !important; transform: translateY(-1px) !important; }
.stButton > button:disabled { background: #111 !important; color: #222 !important; box-shadow: none !important; transform: none !important; }

[data-testid="stDownloadButton"] > button {
    background: transparent !important; color: #22c55e !important;
    border: 1px solid #14532d !important; border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important; font-size: 0.72rem !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    font-weight: 600 !important; width: 100% !important; height: 2.5rem !important;
}
[data-testid="stDownloadButton"] > button:hover { background: #071a0d !important; }

/* Sliders */
[data-testid="stSlider"] > div > div > div > div { background: #22c55e !important; }
[data-testid="stSlider"] label { color: #333 !important; font-size: 0.75rem !important; }

/* Checkboxes */
[data-testid="stCheckbox"] span { color: #333 !important; font-size: 0.78rem !important; }

/* Images */
[data-testid="stImage"] img { border-radius: 12px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #050505; }
::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.8rem 1.4rem 1.2rem;border-bottom:1px solid #0f0f0f;">
        <div style="display:flex;align-items:baseline;gap:0.1em;margin-bottom:0.25rem;">
            <span style="font-family:'Outfit',sans-serif;font-size:1.4rem;font-weight:900;
                color:#fff;letter-spacing:-0.04em;">waste</span>
            <span style="font-family:'Outfit',sans-serif;font-size:1.4rem;font-weight:900;
                color:#22c55e;letter-spacing:-0.04em;">lens</span>
        </div>
        <div style="font-size:0.55rem;color:#1f1f1f;letter-spacing:0.25em;text-transform:uppercase;
            font-weight:600;">AI-powered waste classification</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:1.2rem 1.4rem 0.5rem;">
        <div style="font-size:0.55rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;
            color:#1a1a1a;margin-bottom:0.8rem;">Quick Settings</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 1rem;">', unsafe_allow_html=True)
        conf_thresh = st.slider("Confidence threshold", 0.01, 0.50, 0.25, 0.01)
        iou_thresh = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
        show_labels = st.checkbox("Show labels on image", value=True)
        show_conf_img = st.checkbox("Show confidence score", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:1.2rem 1.4rem 0.5rem;border-top:1px solid #0a0a0a;margin-top:0.5rem;">
        <div style="font-size:0.55rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;
            color:#1a1a1a;margin-bottom:0.8rem;">Detection History</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        if st.button("Clear", key="clear_hist"):
            st.session_state.history = []
            st.rerun()
        cmap = {"recyclable": "#22c55e", "non-recyclable": "#ef4444", "mixed": "#f59e0b", "no-detection": "#333"}
        for i, h in enumerate(reversed(st.session_state.history[-6:])):
            num = len(st.session_state.history) - i
            c = cmap.get(h["verdict"], "#333")
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                padding:0.5rem 1.4rem;border-bottom:1px solid #0a0a0a;">
                <span style="color:#1f1f1f;font-size:0.7rem;">#{num}</span>
                <span style="color:#222;font-size:0.7rem;">{h["count"]} obj</span>
                <span style="color:{c};font-size:0.65rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.1em;">{h["verdict"]}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:0 1.4rem;"><p style="color:#141414;font-size:0.75rem;">No scans yet.</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:1.4rem;border-top:1px solid #0a0a0a;margin-top:auto;">
        <div style="background:#071a0d;border:1px solid #14532d;border-radius:10px;padding:0.8rem 1rem;">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;">
                <div style="width:6px;height:6px;background:#22c55e;border-radius:50%;
                    box-shadow:0 0 6px #22c55e;"></div>
                <span style="color:#22c55e;font-size:0.7rem;font-weight:700;">Model Active</span>
            </div>
            <div style="color:#14532d;font-size:0.65rem;">Vision Pro v4.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN AREA ────────────────────────────────────────────────────
main = st.container()
with main:
    # Top header bar
    st.markdown("""
    <div style="background:#050505;border-bottom:1px solid #0f0f0f;
        padding:1.2rem 2.5rem;display:flex;align-items:center;justify-content:space-between;">
        <div>
            <div style="font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:800;
                color:#fff;letter-spacing:-0.03em;line-height:1.1;">
                Detect. Classify. <span style="color:#22c55e;">Reduce Impact.</span>
            </div>
            <div style="font-size:0.72rem;color:#1f1f1f;margin-top:0.2rem;letter-spacing:0.05em;">
                Upload an image or use your camera to identify waste materials
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:0.8rem;">
            <div style="background:#080808;border:1px solid #111;border-radius:10px;
                padding:0.5rem 1rem;font-size:0.7rem;color:#1f1f1f;font-weight:600;">
                Model: v4.0 (Vision Pro)
            </div>
        </div>
    </div>
    <div style="padding:0 2.5rem;">
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.6], gap="large")

    # ── LEFT: INPUT ─────────────────────────────────────────────
    with col_left:
        st.markdown("""
        <div style="margin-top:1.8rem;">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.25em;text-transform:uppercase;
                color:#1a1a1a;margin-bottom:1rem;">Input source</div>
        </div>
        """, unsafe_allow_html=True)

        mode = st.radio("mode", ["Upload Image", "Use Camera"], horizontal=True, label_visibility="collapsed")

        # Custom styled radio tabs
        st.markdown(f"""
        <div style="display:flex;gap:0.5rem;margin-bottom:1rem;">
            <div style="flex:1;text-align:center;padding:0.6rem;border-radius:10px;
                cursor:pointer;font-size:0.78rem;font-weight:600;
                background:{"#071a0d" if mode=="Upload Image" else "#0a0a0a"};
                color:{"#22c55e" if mode=="Upload Image" else "#222"};
                border:1px solid {"#14532d" if mode=="Upload Image" else "#111"};">
                Upload Image
            </div>
            <div style="flex:1;text-align:center;padding:0.6rem;border-radius:10px;
                cursor:pointer;font-size:0.78rem;font-weight:600;
                background:{"#071a0d" if mode=="Use Camera" else "#0a0a0a"};
                color:{"#22c55e" if mode=="Use Camera" else "#222"};
                border:1px solid {"#14532d" if mode=="Use Camera" else "#111"};">
                Use Camera
            </div>
        </div>
        """, unsafe_allow_html=True)

        source_image = None

        if mode == "Upload Image":
            uploaded = st.file_uploader(
                "Drop image here",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                label_visibility="collapsed"
            )
            if uploaded:
                tmp_path = f"/tmp/wastelens_{uploaded.name}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getvalue())
                source_image = Image.open(tmp_path).convert("RGB")
                st.session_state["img_path"] = tmp_path
                st.markdown('<div style="font-size:0.65rem;color:#1a3a24;margin-top:0.3rem;">Supports: JPG, JPEG, PNG, WEBP, BMP (Max 200MB)</div>', unsafe_allow_html=True)
        else:
            st.caption("Allow camera access, then click the shutter.")
            cam = st.camera_input("cam", label_visibility="collapsed")
            if cam:
                tmp_path = "/tmp/wastelens_camera.jpg"
                with open(tmp_path, "wb") as f:
                    f.write(cam.getvalue())
                source_image = Image.open(tmp_path).convert("RGB")
                st.session_state["img_path"] = tmp_path

        if source_image:
            st.image(source_image, use_column_width=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        run = st.button(
            "+ Analyze Image",
            disabled=(source_image is None),
            use_container_width=True,
            type="primary"
        )

        if source_image is None:
            st.markdown("""
            <div style="text-align:center;margin-top:0.5rem;">
                <span style="font-size:0.7rem;color:#141414;">
                    AI will detect and classify waste in your image
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ── RIGHT: RESULTS ───────────────────────────────────────────
    with col_right:
        st.markdown("""
        <div style="margin-top:1.8rem;margin-bottom:1rem;display:flex;
            align-items:center;justify-content:space-between;">
            <div>
                <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.25em;
                    text-transform:uppercase;color:#1a1a1a;">Live Analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if source_image is None:
            st.markdown("""
            <div style="background:#050505;border:1px solid #0d0d0d;border-radius:18px;
                min-height:400px;display:flex;flex-direction:column;
                align-items:center;justify-content:center;text-align:center;padding:3rem;">
                <div style="width:64px;height:64px;background:#0a0a0a;border-radius:16px;
                    border:1px solid #111;display:flex;align-items:center;
                    justify-content:center;font-size:1.8rem;margin-bottom:1.5rem;
                    color:#1a1a1a;">◈</div>
                <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.25em;
                    text-transform:uppercase;color:#141414;margin-bottom:0.6rem;">
                    Awaiting Input
                </div>
                <div style="font-size:0.8rem;color:#111;line-height:1.7;max-width:240px;">
                    Upload an image or use your camera to begin waste detection
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif run:
            with st.spinner("AI is scanning and analyzing your image..."):
                t0 = time.time()
                img_path = st.session_state.get("img_path")
                predict_input = img_path if img_path and os.path.exists(img_path) else np.array(source_image)
                results = model.predict(predict_input, conf=conf_thresh, iou=iou_thresh, imgsz=640, verbose=False)
                elapsed = time.time() - t0

            boxes = results[0].boxes
            names = results[0].names
            n_det = len(boxes)
            detections = [{"class": names[int(b.cls[0])].lower(), "conf": float(b.conf[0])} for b in boxes]
            RECYCLABLE_CLASSES, FRIENDLY = build_class_maps(model)
            rec = [d for d in detections if d["class"] in RECYCLABLE_CLASSES]
            nonrec = [d for d in detections if d["class"] not in RECYCLABLE_CLASSES]
            verdict = (
                "no-detection" if n_det == 0 else
                "recyclable" if rec and not nonrec else
                "non-recyclable" if nonrec and not rec else
                "mixed"
            )

            # Waste score (simple heuristic)
            if n_det > 0:
                avg_conf = sum(d["conf"] for d in detections) / n_det
                rec_ratio = len(rec) / n_det
                waste_score = int((rec_ratio * 0.6 + avg_conf * 0.4) * 100)
            else:
                waste_score = 0

            annotated_pil = Image.fromarray(results[0].plot(labels=show_labels, conf=show_conf_img))
            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")

            st.session_state.last_result = {
                "annotated_bytes": buf.getvalue(), "annotated_pil": annotated_pil,
                "detections": detections, "verdict": verdict, "elapsed": elapsed,
                "n_det": n_det, "rec_count": len(rec), "nonrec_count": len(nonrec),
                "waste_score": waste_score,
                "avg_conf": int(avg_conf * 100) if n_det > 0 else 0,
            }
            st.session_state.history.append({"count": n_det, "verdict": verdict})

        r = st.session_state.last_result
        if r and source_image is not None:
            # Annotated image
            st.image(r["annotated_pil"], use_column_width=True)

            # Legend dots
            RECYCLABLE_CLASSES, FRIENDLY = build_class_maps(model)
            legend_html = '<div style="display:flex;gap:1rem;margin-top:0.5rem;margin-bottom:1.2rem;flex-wrap:wrap;">'
            seen = set()
            for d in r["detections"]:
                cls = d["class"]
                if cls not in seen:
                    col = CLASS_COLORS.get(cls, "#555")
                    label = FRIENDLY.get(cls, cls.title())
                    legend_html += f'<div style="display:flex;align-items:center;gap:0.4rem;"><div style="width:8px;height:8px;border-radius:50%;background:{col};"></div><span style="font-size:0.7rem;color:#333;">{label}</span></div>'
                    seen.add(cls)
            legend_html += '</div>'
            st.markdown(legend_html, unsafe_allow_html=True)

            # 4 metric cards
            score_color = "#22c55e" if r["waste_score"] >= 60 else "#f59e0b" if r["waste_score"] >= 30 else "#ef4444"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;margin-bottom:1.2rem;">
                <div style="background:#080808;border:1px solid #0f0f0f;border-radius:14px;padding:1rem 0.8rem;text-align:center;">
                    <div style="font-size:0.55rem;color:#1a1a1a;text-transform:uppercase;letter-spacing:0.2em;font-weight:700;margin-bottom:0.4rem;">Waste Score</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:900;color:{score_color};line-height:1;">{r["waste_score"]}</div>
                    <div style="font-size:0.6rem;color:#1a1a1a;margin-top:2px;">/100</div>
                </div>
                <div style="background:#080808;border:1px solid #0f0f0f;border-radius:14px;padding:1rem 0.8rem;text-align:center;">
                    <div style="font-size:0.55rem;color:#1a1a1a;text-transform:uppercase;letter-spacing:0.2em;font-weight:700;margin-bottom:0.4rem;">Total Objects</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:900;color:#fff;line-height:1;">{r["n_det"]}</div>
                    <div style="font-size:0.6rem;color:#1a1a1a;margin-top:2px;">Detected</div>
                </div>
                <div style="background:#080808;border:1px solid #0f0f0f;border-radius:14px;padding:1rem 0.8rem;text-align:center;">
                    <div style="font-size:0.55rem;color:#1a1a1a;text-transform:uppercase;letter-spacing:0.2em;font-weight:700;margin-bottom:0.4rem;">Avg Confidence</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:900;color:#22c55e;line-height:1;">{r["avg_conf"]}%</div>
                    <div style="font-size:0.6rem;color:#1a1a1a;margin-top:2px;">All objects</div>
                </div>
                <div style="background:#080808;border:1px solid #0f0f0f;border-radius:14px;padding:1rem 0.8rem;text-align:center;">
                    <div style="font-size:0.55rem;color:#1a1a1a;text-transform:uppercase;letter-spacing:0.2em;font-weight:700;margin-bottom:0.4rem;">Processing</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:900;color:#333;line-height:1;">{r["elapsed"]*1000:.0f}<span style="font-size:1rem;">ms</span></div>
                    <div style="font-size:0.6rem;color:#1a1a1a;margin-top:2px;">Very Fast</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Bottom two columns: classification breakdown + recommendations
            bc_left, bc_right = st.columns([1, 1], gap="medium")

            with bc_left:
                st.markdown("""
                <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.22em;
                    text-transform:uppercase;color:#1a1a1a;margin-bottom:0.8rem;">
                    Classification Breakdown
                </div>
                """, unsafe_allow_html=True)

                if r["detections"]:
                    # Count per class
                    class_counts = {}
                    for d in r["detections"]:
                        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

                    for cls, count in class_counts.items():
                        is_rec = cls in RECYCLABLE_CLASSES
                        label = FRIENDLY.get(cls, cls.title())
                        col = CLASS_COLORS.get(cls, "#555")
                        pct = int(count / r["n_det"] * 100)
                        avg_c = int(sum(d["conf"] for d in r["detections"] if d["class"] == cls) / count * 100)
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.7rem;">
                            <div style="width:28px;height:28px;border-radius:8px;
                                background:{col}20;border:1px solid {col}40;
                                display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                                <div style="width:8px;height:8px;border-radius:50%;background:{col};"></div>
                            </div>
                            <div style="flex:1;">
                                <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                                    <span style="font-size:0.75rem;font-weight:600;color:#e5e5e5;">{label}</span>
                                    <span style="font-size:0.72rem;color:{col};font-weight:700;">{avg_c}%</span>
                                </div>
                                <div style="background:#0a0a0a;border-radius:4px;height:4px;">
                                    <div style="height:4px;border-radius:4px;background:{col};width:{avg_c}%;"></div>
                                </div>
                                <div style="font-size:0.62rem;color:#1a1a1a;margin-top:2px;">
                                    {count} object{"s" if count>1 else ""} ({pct}%)
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    if r["verdict"] == "no-detection":
                        st.markdown('<p style="color:#1a1a1a;font-size:0.78rem;">Nothing detected. Try lowering the confidence threshold.</p>', unsafe_allow_html=True)

            with bc_right:
                st.markdown("""
                <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.22em;
                    text-transform:uppercase;color:#1a1a1a;margin-bottom:0.8rem;">
                    Recommendations
                </div>
                """, unsafe_allow_html=True)

                # Verdict banner
                v = r["verdict"]
                if v == "recyclable":
                    vbg, vborder, vcolor, vicon, vtitle, vsub = "#071a0d", "#14532d", "#22c55e", "♻", "Recyclable", "Place in the recycling bin"
                elif v == "non-recyclable":
                    vbg, vborder, vcolor, vicon, vtitle, vsub = "#1a0505", "#7f1d1d", "#ef4444", "✕", "Non-Recyclable", "Place in general waste bin"
                elif v == "mixed":
                    vbg, vborder, vcolor, vicon, vtitle, vsub = "#1a1200", "#92400e", "#f59e0b", "⚠", "Mixed Waste", "Separate before disposal"
                else:
                    vbg, vborder, vcolor, vicon, vtitle, vsub = "#0a0a0a", "#111", "#333", "?", "No Detection", "Try a clearer image"

                st.markdown(f"""
                <div style="background:{vbg};border:1px solid {vborder};border-radius:12px;
                    padding:1rem 1.1rem;margin-bottom:0.8rem;display:flex;align-items:center;gap:0.8rem;">
                    <div style="width:36px;height:36px;background:{vbg};border-radius:9px;
                        border:1px solid {vborder};display:flex;align-items:center;
                        justify-content:center;font-size:1.1rem;flex-shrink:0;color:{vcolor};">{vicon}</div>
                    <div>
                        <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.88rem;
                            color:{vcolor};">{vtitle}</div>
                        <div style="font-size:0.68rem;color:{vborder};margin-top:1px;">{vsub}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Tips per detected class
                seen_tips = set()
                for d in r["detections"]:
                    cls = d["class"]
                    if cls not in seen_tips:
                        tip = DISPOSAL_TIPS.get(cls, DEFAULT_TIP)
                        label = FRIENDLY.get(cls, cls.title())
                        col = CLASS_COLORS.get(cls, "#555")
                        st.markdown(f"""
                        <div style="display:flex;gap:0.6rem;align-items:flex-start;margin-bottom:0.6rem;">
                            <div style="width:18px;height:18px;border-radius:5px;background:{col}20;
                                border:1px solid {col}40;display:flex;align-items:center;
                                justify-content:center;flex-shrink:0;margin-top:1px;">
                                <div style="width:5px;height:5px;border-radius:50%;background:{col};"></div>
                            </div>
                            <div style="font-size:0.72rem;color:#252525;line-height:1.6;">
                                <span style="color:{col};font-weight:600;">{label}:</span> {tip}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        seen_tips.add(cls)

                st.markdown("<div style='margin-top:0.8rem;'>", unsafe_allow_html=True)
                st.download_button(
                    "Download Report",
                    data=r["annotated_bytes"],
                    file_name="wastelens_result.png",
                    mime="image/png",
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
