import os
import uuid
import tempfile
from PIL import Image, ImageOps
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
import cv2

app = Flask(__name__)

MODEL_PATH = "waste_final_best.pt"
FIXED_CONFIDENCE = 0.25

model = YOLO(MODEL_PATH)

UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SUPPORTED_CLASSES = ["cans", "foodwaste", "glass", "paperwaste", "plasticbottles"]


def run_detection(image_path: str):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        normalized_path = tmp.name
        image.save(normalized_path, format="JPEG", quality=95)

    try:
        results = model.predict(
            source=normalized_path,
            conf=FIXED_CONFIDENCE,
            imgsz=640,
            verbose=False
        )

        result = results[0]
        plotted = result.plot()

        result_filename = f"{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(RESULT_DIR, result_filename)
        cv2.imwrite(result_path, plotted)

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                detections.append((class_name, conf))

        detections.sort(key=lambda x: x[1], reverse=True)
        return result_filename, detections

    finally:
        if os.path.exists(normalized_path):
            os.remove(normalized_path)


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        uploaded_image=None,
        result_image=None,
        detections=[],
        detection_count=0,
        top_class="—",
        top_conf="—",
        status="Awaiting image input.",
        confidence=FIXED_CONFIDENCE,
        supported_classes=SUPPORTED_CLASSES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template(
            "index.html",
            uploaded_image=None,
            result_image=None,
            detections=[],
            detection_count=0,
            top_class="—",
            top_conf="—",
            status="No file was uploaded.",
            confidence=FIXED_CONFIDENCE,
            supported_classes=SUPPORTED_CLASSES,
        )

    file = request.files["file"]
    if file.filename == "":
        return render_template(
            "index.html",
            uploaded_image=None,
            result_image=None,
            detections=[],
            detection_count=0,
            top_class="—",
            top_conf="—",
            status="No file was selected.",
            confidence=FIXED_CONFIDENCE,
            supported_classes=SUPPORTED_CLASSES,
        )

    ext = os.path.splitext(file.filename)[1].lower()
    upload_filename = f"{uuid.uuid4().hex}{ext}"
    upload_path = os.path.join(UPLOAD_DIR, upload_filename)
    file.save(upload_path)

    try:
        result_filename, detections = run_detection(upload_path)

        detection_count = len(detections)
        top_class = detections[0][0] if detections else "None"
        top_conf = f"{detections[0][1]:.0%}" if detections else "—"

        status = (
            f"Analysis complete — {detection_count} object(s) identified."
            if detections
            else f"No objects detected at confidence threshold {FIXED_CONFIDENCE}."
        )

        formatted_detections = [
            {
                "label": label,
                "confidence": f"{conf:.0%}",
                "confidence_raw": conf,
            }
            for label, conf in detections
        ]

        return render_template(
            "index.html",
            uploaded_image=url_for("static", filename=f"uploads/{upload_filename}"),
            result_image=url_for("static", filename=f"results/{result_filename}"),
            detections=formatted_detections,
            detection_count=detection_count,
            top_class=top_class,
            top_conf=top_conf,
            status=status,
            confidence=FIXED_CONFIDENCE,
            supported_classes=SUPPORTED_CLASSES,
        )

    except Exception as e:
        return render_template(
            "index.html",
            uploaded_image=url_for("static", filename=f"uploads/{upload_filename}"),
            result_image=None,
            detections=[],
            detection_count=0,
            top_class="Error",
            top_conf="—",
            status=f"Inference failed: {str(e)}",
            confidence=FIXED_CONFIDENCE,
            supported_classes=SUPPORTED_CLASSES,
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
