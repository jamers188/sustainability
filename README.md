# WasteLens — AI-Powered Waste Classification

> Point your camera or upload a photo. WasteLens identifies waste materials in real time and tells you exactly how to dispose of them.

---

## What it does

WasteLens uses a custom-trained YOLOv8 object detection model to identify waste materials in images and classify them as recyclable or non-recyclable. It provides disposal tips for each detected item and a confidence score for every detection.

**Detectable classes:**
- Plastic Bottles
- Glass
- Cans
- Paper / Cardboard
- Food Waste

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend + Backend | Python · Streamlit |
| Object Detection | YOLOv8s (Ultralytics) |
| Model Training | Google Colab · T4 GPU |
| Dataset | Custom unified dataset (5 classes, ~2,400 images) + Roboflow public datasets |
| Deployment | Streamlit Cloud |
| Model Hosting | Google Drive (downloaded at runtime) |

---

## Project Structure

```
sustainability/
├── app.py                  # Main application (frontend + backend)
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level dependencies
├── .python-version         # Python version pin (3.11)
├── .streamlit/
│   └── config.toml         # Streamlit dark theme config
└── README.md
```

---

## How It Works

```
User uploads image
        ↓
Streamlit handles the UI and triggers inference
        ↓
YOLOv8s model runs object detection (conf=0.25)
        ↓
Detections classified as Recyclable / Non-Recyclable
        ↓
Results displayed with confidence scores and disposal tips
```

The model file is hosted on Google Drive and downloaded automatically on first run. This avoids GitHub's file size limits for large binary files.

---

## Model Training

The model was trained on a unified dataset combining:
- Original labeled data (490 images per class for cans and plastic bottles, 420 for others)
- Additional Roboflow public datasets for plastic bottles, glass, cardboard, and food waste
- Balanced at 600–800 images per class to avoid class bias

**Training config:**
```python
model = YOLO("yolov8s.pt")
model.train(
    epochs=150,
    imgsz=640,
    batch=16,
    patience=30,
    mosaic=1.0,
    mixup=0.1,
)
```

**Final model performance (mAP@0.5):**
| Class | mAP@0.5 |
|---|---|
| Plastic Bottles | 0.987 |
| Cans | 0.934 |
| Glass | 0.886 |
| Paper Waste | 0.678 |
| Food Waste | 0.536 |
| **Overall** | **0.804** |

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/your-username/sustainability.git
cd sustainability

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

