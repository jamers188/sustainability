import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import io
import cv2

# ─────────────────────────────────────────
#  PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────
st.set_page_config(
    page_title="WasteLens — AI Waste Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Reset & base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e0f11;
    color: #e8e6e1;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 2rem 4rem; max-width: 1200px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #13151a !important;
    border-right: 1px solid #252830;
  }
  [data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5a9e6f;
    border-bottom: 1px solid #252830;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }

  /* ── Page header ── */
  .wl-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    border-bottom: 1px solid #252830;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
  }
  .wl-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin: 0;
  }
  .wl-header .tagline {
    font-size: 0.85rem;
    color: #6b7280;
    font-weight: 300;
  }
  .wl-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    background: #1a2e22;
    color: #5a9e6f;
    border: 1px solid #2d4a38;
    padding: 2px 8px;
    border-radius: 2px;
  }

  /* ── Cards ── */
  .wl-card {
    background: #13151a;
    border: 1px solid #252830;
    border-radius: 6px;
    padding: 1.4rem;
    margin-bottom: 1rem;
  }
  .wl-card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 1rem;
  }

  /* ── Result pill ── */
  .result-recyclable {
    background: #1a2e22;
    border: 1px solid #2d4a38;
    border-left: 3px solid #5a9e6f;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    color: #7ed4a0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
  }
  .result-nonrecyclable {
    background: #2a1a1a;
    border: 1px solid #4a2d2d;
    border-left: 3px solid #c05050;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    color: #e07070;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
  }
  .result-mixed {
    background: #1e1e10;
    border: 1px solid #3a3520;
    border-left: 3px solid #c0a040;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    color: #d4b86a;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
  }

  /* ── Detection tags ── */
  .tag-recyclable {
    display: inline-block;
    background: #1a2e22;
    color: #7ed4a0;
    border: 1px solid #2d4a38;
    border-radius: 3px;
    padding: 2px 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    margin: 2px;
  }
  .tag-nonrecyclable {
    display: inline-block;
    background: #2a1a1a;
    color: #e07070;
    border: 1px solid #4a2d2d;
    border-radius: 3px;
    padding: 2px 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    margin: 2px;
  }

  /* ── Metric row ── */
  .metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
  }
  .metric-box {
    flex: 1;
    background: #0e0f11;
    border: 1px solid #252830;
    border-radius: 4px;
    padding: 0.9rem 1rem;
    text-align: center;
  }
  .metric-box .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
  }
  .metric-box .lbl {
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
  }

  /* ── Confidence bar ── */
  .conf-bar-bg {
    background: #252830;
    border-radius: 2px;
    height: 5px;
    margin-top: 4px;
  }
  .conf-bar-fill {
    height: 5px;
    border-radius: 2px;
    background: #5a9e6f;
  }

  /* ── History table ── */
  .hist-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1c1e24;
    font-size: 0.82rem;
  }
  .hist-row:last-child { border-bottom: none; }

  /* ── Tip box ── */
  .tip-box {
    background: #0e1218;
    border: 1px dashed #252830;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #6b7280;
    line-height: 1.6;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploadDropzone"] {
    background: #0e0f11 !important;
    border: 1px dashed #333640 !important;
    border-radius: 6px !important;
    color: #6b7280 !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: #1a2e22 !important;
    color: #7ed4a0 !important;
    border: 1px solid #2d4a38 !important;
    border-radius: 3px !important;
    padding: 0.4rem 1.2rem !important;
    transition: all 0.15s ease !important;
  }
  .stButton > button:hover {
    background: #2d4a38 !important;
    border-color: #5a9e6f !important;
  }

  /* ── Slider ── */
  [data-testid="stSlider"] .stSlider > div > div {
    background: #5a9e6f !important;
  }

  /* ── Divider ── */
  hr { border-color: #252830 !important; }

  /* ── Radio ── */
  [data-testid="stRadio"] label { font-size: 0.85rem !important; }

  /* ── Info / warning / error ── */
  [data-testid="stAlert"] {
    background: #13151a !important;
    border-color: #252830 !important;
    border-radius: 4px !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────
RECYCLABLE_CLASSES = {"cans", "glass", "paperwaste", "plasticbottles"}

DISPOSAL_TIPS = {
    "cans":           "Rinse cans before recycling. Crush to save space.",
    "glass":          "Remove lids. Sort by color if your facility requires it.",
    "paperwaste":     "Keep dry. Remove staples and plastic windows from envelopes.",
    "plasticbottles": "Empty and rinse. Check the resin code (1-7) on your bottle.",
    "default":        "Bag securely in black bin liner. Do not mix with recyclables.",
}

FRIENDLY_NAMES = {
    "cans":           "Metal Cans",
    "glass":          "Glass",
    "paperwaste":     "Paper / Cardboard",
    "plasticbottles": "Plastic Bottles",
}


# ─────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("waste_final_best.pt")

model = load_model()


# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts


# ─────────────────────────────────────────
#  SIDEBAR — SETTINGS & HISTORY
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")

    conf_thresh = st.slider(
        "Detection confidence threshold",
        min_value=0.05, max_value=0.90, value=0.25, step=0.05,
        help="Lower = more detections but more false positives"
    )

    iou_thresh = st.slider(
        "NMS IoU threshold",
        min_value=0.1, max_value=0.9, value=0.45, step=0.05,
        help="Controls duplicate box suppression"
    )

    show_labels = st.checkbox("Show labels on image", value=True)
    show_conf_on_image = st.checkbox("Show confidence on image", value=True)

    st.markdown("---")
    st.markdown("### Detection History")

    if st.session_state.history:
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            verdict_color = (
                "#7ed4a0" if h["verdict"] == "recyclable"
                else "#e07070" if h["verdict"] == "non-recyclable"
                else "#d4b86a"
            )
            st.markdown(
                f'<div class="hist-row">'
                f'<span style="color:#9ca3af;font-family:Space Mono,monospace;font-size:0.65rem;">#{len(st.session_state.history)-i}</span>'
                f'<span style="color:#e8e6e1">{h["count"]} object{"s" if h["count"]!=1 else ""}</span>'
                f'<span style="color:{verdict_color};font-family:Space Mono,monospace;font-size:0.7rem;">{h["verdict"].upper()}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<p style="color:#6b7280;font-size:0.8rem;">No scans yet.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div class="tip-box">Model: <b>waste_final_best.pt</b><br>'
        'Classes: cans · glass · paper · plastic bottles</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────
st.markdown("""
<div class="wl-header">
  <h1>WasteLens</h1>
  <span class="wl-badge">AI Classifier</span>
  <span class="tagline">Point, detect, sort correctly.</span>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT: INPUT ──────────────────────────
with left_col:
    st.markdown('<div class="wl-card">', unsafe_allow_html=True)
    st.markdown('<div class="wl-card-title">Input source</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input source",
        ["Upload image", "Use camera"],
        label_visibility="collapsed",
        horizontal=True,
    )

    source_image = None

    if input_mode == "Upload image":
        uploaded = st.file_uploader(
            "Drop an image here or click to browse",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded:
            source_image = Image.open(uploaded).convert("RGB")

    else:  # camera
        st.markdown(
            '<p style="font-size:0.8rem;color:#9ca3af;margin-bottom:0.5rem;">'
            'Allow camera access in your browser, then click the shutter button below.</p>',
            unsafe_allow_html=True,
        )
        cam = st.camera_input("Camera", label_visibility="collapsed")
        if cam:
            source_image = Image.open(cam).convert("RGB")

    st.markdown('</div>', unsafe_allow_html=True)

    if source_image:
        st.image(source_image, caption="Original image", use_column_width=True)

    # Run button
    run_disabled = source_image is None
    run = st.button("Analyse image", disabled=run_disabled, use_container_width=True)


# ── RIGHT: RESULTS ───────────────────────
with right_col:
    if source_image is None:
        st.markdown("""
        <div class="wl-card" style="min-height:260px;display:flex;flex-direction:column;
             justify-content:center;align-items:center;text-align:center;">
          <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
               letter-spacing:0.2em;text-transform:uppercase;color:#333640;margin-bottom:0.8rem;">
            Awaiting scan
          </div>
          <div style="color:#252830;font-size:3rem;margin-bottom:0.5rem;">[ ]</div>
          <div style="font-size:0.82rem;color:#4b5563;">
            Upload or capture an image on the left,<br>then press Analyse.
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif run or ("last_result" in st.session_state and not run):
        if run and source_image:
            with st.spinner("Running detection..."):
                t0 = time.time()
                img_array = np.array(source_image)
                results = model.predict(
                    img_array,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    verbose=False,
                )
                elapsed = time.time() - t0

            boxes  = results[0].boxes
            names  = results[0].names
            n_det  = len(boxes)

            detections = []
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id].lower()
                conf_val = float(box.conf[0])
                detections.append({"class": cls_name, "conf": conf_val})

            # Annotated image
            annotated = results[0].plot(
                labels=show_labels,
                conf=show_conf_on_image,
            )
            annotated_pil = Image.fromarray(annotated)

            # Determine verdict
            if n_det == 0:
                verdict = "no-detection"
            else:
                rec = [d for d in detections if d["class"] in RECYCLABLE_CLASSES]
                nonrec = [d for d in detections if d["class"] not in RECYCLABLE_CLASSES]
                if rec and not nonrec:
                    verdict = "recyclable"
                elif nonrec and not rec:
                    verdict = "non-recyclable"
                else:
                    verdict = "mixed"

            # Store
            st.session_state.last_result = {
                "annotated": annotated_pil,
                "detections": detections,
                "verdict": verdict,
                "elapsed": elapsed,
                "n_det": n_det,
            }
            st.session_state.history.append({
                "count": n_det,
                "verdict": verdict,
            })

        # ── Display stored result ──
        r = st.session_state.get("last_result")
        if r:
            # Annotated image
            st.image(r["annotated"], caption="Detection overlay", use_column_width=True)

            # Metrics
            rec_count = sum(1 for d in r["detections"] if d["class"] in RECYCLABLE_CLASSES)
            nonrec_count = r["n_det"] - rec_count
            st.markdown(
                f'<div class="metric-row">'
                f'<div class="metric-box"><div class="val">{r["n_det"]}</div><div class="lbl">Objects found</div></div>'
                f'<div class="metric-box"><div class="val" style="color:#7ed4a0">{rec_count}</div><div class="lbl">Recyclable</div></div>'
                f'<div class="metric-box"><div class="val" style="color:#e07070">{nonrec_count}</div><div class="lbl">Non-recyclable</div></div>'
                f'<div class="metric-box"><div class="val" style="color:#9ca3af">{r["elapsed"]*1000:.0f}ms</div><div class="lbl">Inference</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Verdict banner
            if r["verdict"] == "recyclable":
                st.markdown('<div class="result-recyclable">RECYCLABLE — Place in the recycling bin.</div>', unsafe_allow_html=True)
            elif r["verdict"] == "non-recyclable":
                st.markdown('<div class="result-nonrecyclable">NON-RECYCLABLE — General waste bin.</div>', unsafe_allow_html=True)
            elif r["verdict"] == "mixed":
                st.markdown('<div class="result-mixed">MIXED — Separate items before disposal.</div>', unsafe_allow_html=True)
            else:
                st.warning("Nothing detected. Try lowering the confidence threshold in the sidebar.")

            # Per-detection breakdown
            if r["detections"]:
                st.markdown('<div class="wl-card" style="margin-top:1rem;">', unsafe_allow_html=True)
                st.markdown('<div class="wl-card-title">Detection breakdown</div>', unsafe_allow_html=True)

                for d in r["detections"]:
                    is_rec = d["class"] in RECYCLABLE_CLASSES
                    friendly = FRIENDLY_NAMES.get(d["class"], d["class"].title())
                    tag_cls = "tag-recyclable" if is_rec else "tag-nonrecyclable"
                    label = "Recyclable" if is_rec else "Non-recyclable"
                    tip = DISPOSAL_TIPS.get(d["class"], DISPOSAL_TIPS["default"])
                    st.markdown(
                        f'<div style="margin-bottom:0.8rem;">'
                        f'<span class="{tag_cls}">{friendly}</span>'
                        f'<span style="font-family:Space Mono,monospace;font-size:0.65rem;'
                        f'color:#6b7280;margin-left:0.5rem;">{d["conf"]*100:.1f}% conf</span>'
                        f'<div class="conf-bar-bg"><div class="conf-bar-fill" '
                        f'style="width:{d["conf"]*100:.1f}%;'
                        f'background:{"#5a9e6f" if is_rec else "#c05050"};"></div></div>'
                        f'<div style="font-size:0.75rem;color:#9ca3af;margin-top:4px;">{tip}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('</div>', unsafe_allow_html=True)

            # Download annotated image
            buf = io.BytesIO()
            r["annotated"].save(buf, format="PNG")
            st.download_button(
                label="Download annotated image",
                data=buf.getvalue(),
                file_name="wastelens_result.png",
                mime="image/png",
                use_container_width=True,
            )
