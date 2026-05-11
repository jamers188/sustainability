# WasteLens — AI-Powered Waste Classification

Point your camera or upload a photo. WasteLens identifies waste materials in real time and tells you exactly how to dispose of them.

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
| Dataset | Custom unified dataset (5 classes, ~2,965 images) + Roboflow public datasets |
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
├── dataset.ipynb           # Dataset preparation and food waste expansion
├── EXP2_TRAINING.ipynb     # Training and evaluation scripts
├── data.yaml               # Dataset configuration
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

### Project 1 — Baseline Model
The baseline model was trained on a unified dataset combining:
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

**Baseline performance (mAP@0.5):**

| Class           | mAP@0.5 |
|-----------------|---------|
| Plastic Bottles | 0.987   |
| Cans            | 0.934   |
| Glass           | 0.886   |
| Paper Waste     | 0.678   |
| Food Waste      | 0.536   |
| **Overall**     | **0.804** |

---

### Project 2 — Improved Model
Two improvement strategies were applied to address weaknesses identified in the baseline, particularly low food waste recall (0.525) and paper waste recall (0.602):

- **Experiment 1** — Backbone upgrade: YOLOv8n → YOLOv8s (Category A — Model Architecture)
- **Experiment 2** — Dataset expansion: +160 food waste images from Roboflow (Category B — Dataset Enhancement)
- **Final Model** — Combined: YOLOv8s + expanded dataset

**Training config (all experiments):**
```python
model = YOLO("yolov8s.pt")  # or yolov8n.pt for Exp2
model.train(
    data="unified_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    seed=42,
    workers=4,
    cache=True,
)
```

**Consolidated results:**

| Model              | Precision | Recall | mAP50 | mAP50-95 |
|--------------------|-----------|--------|-------|----------|
| Baseline (YOLOv8n) | 0.856     | 0.722  | 0.722 | 0.531    |
| Exp1 (YOLOv8s)     | 0.844     | 0.742  | 0.798 | 0.572    |
| Exp2 (Expanded)    | 0.830     | 0.756  | 0.796 | 0.577    |
| Final (Combined)   | 0.828     | 0.761  | 0.792 | 0.570    |

**Per-class mAP50 (Final Combined Model):**

| Class           | Baseline | Final  | Change  |
|-----------------|----------|--------|---------|
| Cans            | 0.909    | 0.945  | +0.036  |
| Food Waste      | 0.545    | 0.613  | +0.068  |
| Glass           | 0.767    | 0.844  | +0.077  |
| Paper Waste     | 0.610    | 0.650  | +0.040  |
| Plastic Bottles | 0.777    | 0.907  | +0.130  |
| **Overall**     | **0.722**| **0.792** | **+0.070** |

---

## 🔗 Trained Model Weights
Download the final best model weights here:
**[best.pt — Google Drive](https://drive.google.com/drive/folders/12BaI2madSIEg7-_GxImnkQ3d0fE9HbTM?usp=sharing)**

---

## Dataset Structure (Project 2)

```
unified_dataset/
├── train/
│   ├── images/    (2,965 images after expansion)
│   └── labels/
├── valid/
│   ├── images/    (832 images)
│   └── labels/
├── test/
│   ├── images/    (420 images)
│   └── labels/
└── data.yaml
```

**Class distribution after food waste expansion:**

| Class           | Train | Valid | Test | Total |
|-----------------|-------|-------|------|-------|
| Cans            | 632   | 178   | 90   | 900   |
| Food Waste      | 653   | 174   | 90   | 917   |
| Glass           | 560   | 160   | 80   | 800   |
| Paper Waste     | 560   | 160   | 80   | 800   |
| Plastic Bottles | 560   | 160   | 80   | 800   |

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

---

## Reproducing the Training (Project 2)

### Step 1 — Mount Drive and install
```python
from google.colab import drive
drive.mount('/content/drive')
!pip install ultralytics
```

### Step 2 — Copy dataset to local storage
```python
import shutil
shutil.copytree(
    "/content/drive/MyDrive/waste_project/unified_dataset",
    "/content/unified_dataset"
)
```

### Step 3 — Train Experiment 1 (backbone upgrade)
```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(
    data="/content/unified_dataset/data.yaml",
    epochs=100, imgsz=640, batch=16, seed=42,
    workers=4, cache=True,
    project="/content/drive/MyDrive/waste_project",
    name="exp1_yolov8s"
)
```

### Step 4 — Train Experiment 2 (expanded dataset)
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="/content/unified_dataset/data.yaml",
    epochs=100, imgsz=640, batch=16, seed=42,
    workers=4, cache=True,
    project="/content/drive/MyDrive/waste_project",
    name="exp2_foodwaste_expanded"
)
```

### Step 5 — Train Final Combined model
```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(
    data="/content/unified_dataset/data.yaml",
    epochs=100, imgsz=640, batch=16, seed=42,
    workers=4, cache=True,
    project="/content/drive/MyDrive/waste_project",
    name="final_combined"
)
```

### Step 6 — Evaluate all models
```python
from ultralytics import YOLO

for exp, path in [
    ("Exp1",  "exp1_yolov8s/weights/best.pt"),
    ("Exp2",  "exp2_foodwaste_expanded/weights/best.pt"),
    ("Final", "final_combined/weights/best.pt"),
]:
    model = YOLO(f"/content/drive/MyDrive/waste_project/{path}")
    metrics = model.val(
        data="/content/unified_dataset/data.yaml",
        split="test", seed=42, plots=True
    )
    print(f"\n{exp} → mAP50: {metrics.box.map50:.3f} | Recall: {metrics.box.mr:.3f}")
```

### Step 7 — Test on a custom image
```python
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("/content/drive/MyDrive/waste_project/final_combined/weights/best.pt")
results = model("your_image.jpg", conf=0.25)
result_img = results[0].plot()
plt.imshow(result_img[..., ::-1])
plt.axis('off')
plt.show()
```

---

## Environment

| Component   | Version         |
|-------------|-----------------|
| Python      | 3.12.13         |
| PyTorch     | 2.10.0+cu128    |
| Ultralytics | 8.4.48          |
| CUDA        | 12.8            |
| GPU         | NVIDIA Tesla T4 |
| Platform    | Google Colab    |

---

## ⚠️ Ethical Notes
- No identifiable faces, licence plates, or personal data in any dataset
- External food waste images sourced from Roboflow (CC BY 4.0 licence)
- Model confidence scores displayed to users for all detections
- Food waste recall (0.63) is insufficient for fully autonomous sorting without human oversight
- Random seed 42 fixed across all experiments for full reproducibility
