import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Waste Classifier", layout="centered")

# Load your model (cached so it only loads once)
@st.cache_resource
def load_model():
    return YOLO("waste_final_best.pt")

model = load_model()

# Recyclability Logic
RECYCLABLE_CLASSES = ["cans", "glass", "paperwaste", "plasticbottles"]

# --- UI ---
st.title("♻️ Smart Waste Classifier")
st.write("Upload a photo of waste to see if it's recyclable.")

# Choice between Upload and Camera
option = st.radio("Select Input Method:", ("Upload Image", "Use Camera"))

if option == "Upload Image":
    source = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:
    source = st.camera_input("Take a picture of the waste")

# --- INFERENCE ---
if source is not None:
    # Convert file to PIL Image
    img = Image.open(source)
    
    # Run YOLOv8
    results = model.predict(img, conf=0.25)
    
    # UI Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Analysis")
        # Plot results on the image
        res_plotted = results[0].plot()
        st.image(res_plotted, channels="BGR", use_column_width=True)

    with col2:
        st.subheader("Result")
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                is_recyclable = label in RECYCLABLE_CLASSES
                
                if is_recyclable:
                    st.success(f"**{label.upper()}** ({conf:.1%})")
                    st.info("✅ This is **Recyclable**!")
                else:
                    st.warning(f"**{label.upper()}** ({conf:.1%})")
                    st.error("❌ This is **Non-Recyclable**.")
        else:
            st.info("No waste detected. Try a clearer angle!")
