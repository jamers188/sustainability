import os
import tempfile
from PIL import Image, ImageOps
import streamlit as st
from ultralytics import YOLO

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Campus Sustainability Monitoring",
    page_icon="♻️",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #0c0f14 0%, #121821 100%);
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .hero {
        background: linear-gradient(135deg, rgba(61,220,132,0.16), rgba(61,220,132,0.04));
        border: 1px solid rgba(61,220,132,0.18);
        border-radius: 22px;
        padding: 28px 30px;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f5f7fa;
        margin-bottom: 0.4rem;
    }

    .hero-sub {
        font-size: 1rem;
        color: #b8c2cc;
        line-height: 1.7;
        margin-bottom: 1rem;
    }

    .badge-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 8px;
    }

    .badge {
        background: rgba(61,220,132,0.10);
        color: #87efb1;
        border: 1px solid rgba(61,220,132,0.18);
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 600;
    }

    .panel {
        background: #151b24;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
    }

    .panel-title {
        font-size: 1rem;
        font-weight: 700;
        color: #f5f7fa;
        margin-bottom: 12px;
    }

    .metric-box {
        background: #111720;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
    }

    .metric-label {
        color: #92a0ad;
        font-size: 0.8rem;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .metric-value {
        color: #3ddc84;
        font-size: 1.6rem;
        font-weight: 800;
    }

    .info-box {
        background: #111720;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 14px;
        color: #b8c2cc;
        line-height: 1.7;
        font-size: 0.95rem;
    }

    .small-note {
        color: #8f9aa6;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
MODEL_PATH = "waste_final_best.pt"
FIXED_CONFIDENCE = 0.25

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------------------------------
# HERO
# -------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">♻️ Campus Sustainability Monitoring</div>
    <div class="hero-sub">
        AI-powered waste detection system using a custom YOLOv8 model for identifying campus waste categories
        such as cans, food waste, glass, paper waste, and plastic bottles.
    </div>
    <div class="badge-row">
        <div class="badge">YOLOv8 Model</div>
        <div class="badge">Fixed Confidence: 0.25</div>
        <div class="badge">5 Waste Classes</div>
        <div class="badge">Real-Time Inference</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
left_col, right_col = st.columns([2.15, 1], gap="large")

with right_col:
    st.markdown('<div class="panel"><div class="panel-title">Supported Classes</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        • Cans<br>
        • Food Waste<br>
        • Glass<br>
        • Paper Waste<br>
        • Plastic Bottles
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Detection Settings</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <strong>Confidence Threshold:</strong> {FIXED_CONFIDENCE}<br>
        <strong>Input Size:</strong> 640 × 640<br>
        <strong>Inference Mode:</strong> Object Detection
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Best Results Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Use clear, well-lit images with the object fully visible. Avoid heavy blur,
        extreme angles, or very small objects in the frame.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with left_col:
    st.markdown('<div class="panel"><div class="panel-title">Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    run_detection = st.button("Run Detection", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="panel"><div class="panel-title">Input Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if run_detection:
            with st.spinner("Running detection..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    temp_path = tmp.name
                    image.save(temp_path, format="JPEG", quality=95)

                try:
                    results = model.predict(
                        source=temp_path,
                        conf=FIXED_CONFIDENCE,
                        imgsz=640,
                        verbose=False
                    )

                    result = results[0]
                    plotted = result.plot()
                    plotted_rgb = plotted[:, :, ::-1]

                    with col2:
                        st.markdown('<div class="panel"><div class="panel-title">Detection Result</div>', unsafe_allow_html=True)
                        st.image(plotted_rgb, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    if result.boxes is not None and len(result.boxes) > 0:
                        detections = []
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = model.names[cls_id]
                            detections.append((class_name, conf))

                        detections.sort(key=lambda x: x[1], reverse=True)
                        top_class, top_conf = detections[0]

                        m1, m2, m3 = st.columns(3)

                        with m1:
                            st.markdown("""
                            <div class="metric-box">
                                <div class="metric-label">Detections</div>
                                <div class="metric-value">{}</div>
                            </div>
                            """.format(len(detections)), unsafe_allow_html=True)

                        with m2:
                            st.markdown("""
                            <div class="metric-box">
                                <div class="metric-label">Top Class</div>
                                <div class="metric-value" style="font-size:1.1rem;">{}</div>
                            </div>
                            """.format(top_class), unsafe_allow_html=True)

                        with m3:
                            st.markdown("""
                            <div class="metric-box">
                                <div class="metric-label">Top Confidence</div>
                                <div class="metric-value">{:.0%}</div>
                            </div>
                            """.format(top_conf), unsafe_allow_html=True)

                        st.markdown('<div class="panel"><div class="panel-title">Detection Details</div>', unsafe_allow_html=True)

                        detail_lines = []
                        for idx, (cls_name, conf) in enumerate(detections, start=1):
                            detail_lines.append(f"{idx}. {cls_name} — {conf:.2f}")

                        st.text("\n".join(detail_lines))
                        st.markdown('</div>', unsafe_allow_html=True)

                    else:
                        st.warning("No objects detected at confidence threshold 0.25.")

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    else:
        st.markdown('<div class="small-note">Upload an image and click <strong>Run Detection</strong> to begin.</div>', unsafe_allow_html=True)
