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
    img = Image.open(source)
    
    # Convert PIL to numpy array
    img_array = np.array(img)
    
    # Run prediction
    results = model.predict(img_array, conf=0.25)
    
    # Debugging: Show in the app how many things were found
    num_found = len(results[0].boxes)
    st.write(f"Debug: Found {num_found} objects")

    if num_found > 0:
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detected Waste", use_column_width=True)
    else:
        st.warning("Still nothing? Try lowering the 'conf' parameter in your code to 0.1.")
 
