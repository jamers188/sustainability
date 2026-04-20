import os

streamlit_code = '''
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# --- Configuration ---
# Path to your trained YOLOv8 model. Make sure this path is correct for your environment.
# If running locally, you'll likely need to place the model file in the same directory
# as your app.py or provide a full path to it.
MODEL_PATH = "waste_final_v2_best.pt" # Assuming model is in the same directory as app.py
# If model is in a fixed path (e.g., in a specific folder on your drive)
# MODEL_PATH = "/path/to/your/model/waste_final_v2_best.pt"

# Recyclability Logic
RECYCLABLE_CLASSES = ["cans", "glass", "paperwaste", "plasticbottles"]

# --- Streamlit App ---
st.set_page_config(page_title="Smart Waste Classifier", layout="wide", icon="♻️")

st.title("♻️ Smart Waste Classifier")
st.write("Upload an image or use your camera to detect different types of waste and check their recyclability.")

# Sidebar for settings
st.sidebar.header("App Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
st.sidebar.info("Adjust the confidence threshold to filter detection results. Lower values may show more detections, including false positives.")

@st.cache_resource
def load_model():
    """Loads the YOLOv8 model and caches it."""
    try:
        # Ensure the model path is absolute if running in Colab and model is in Drive
        model = YOLO("/content/drive/MyDrive/waste_project/improved_model/waste_final_v2_best.pt") # Fixed path for Colab
        st.sidebar.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}. Please ensure the model path is correct and accessible.")
        return None

model = load_model()

# --- Input Method Selection ---
input_option = st.radio("Select Input Method:", ("Upload Image", "Use Camera"))

source_image = None
if input_option == "Upload Image":
    source_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
elif input_option == "Use Camera":
    st.warning("Camera input might not work in all deployment environments (e.g., Colab or certain cloud platforms). It generally works best when running Streamlit locally.")
    source_image = st.camera_input("Take a picture of the waste")

# --- Inference and Display Results ---
if source_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded/Captured Image")
        image = Image.open(source_image)
        st.image(image, caption='Input Image', use_column_width=True)

    st.info("Detecting objects...")

    if model:
        img_np = np.array(image)
        results = model.predict(source=img_np, conf=confidence_threshold)

        if results and len(results[0].boxes) > 0:
            with col2:
                st.subheader("Detection Results")
                annotated_image = results[0].plot()
                annotated_image_rgb = Image.fromarray(annotated_image[..., ::-1]) # Convert BGR to RGB
                st.image(annotated_image_rgb, caption='Annotated Image', use_column_width=True)

            st.subheader("Detected Objects Summary:")
            detected_data = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls] if hasattr(model, 'names') and cls < len(model.names) else f"Class {cls}"

                    is_recyclable = "Yes" if class_name in RECYCLABLE_CLASSES else "No"
                    detected_data.append({"Object": class_name, "Confidence": f"{conf:.2f}", "Recyclable": is_recyclable})

            df_detections = pd.DataFrame(detected_data)
            st.table(df_detections)

            # Provide general recyclability feedback
            recyclable_found = any(item["Recyclable"] == "Yes" for item in detected_data)
            if recyclable_found:
                st.success("Some recyclable items were detected! Please sort them accordingly.")
            else:
                st.warning("No easily recyclable items were detected from the predefined categories. Please check local guidelines.")

        else:
            with col2:
                st.subheader("No Objects Detected")
                st.warning("No objects were detected in the image with the current confidence threshold. Try adjusting the slider in the sidebar.")
            st.image(image, caption='Original Image (No Detections)', use_column_width=True)
    else:
        st.error("Model could not be loaded. Please check the model path.")

# Footer or additional info
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:** This app uses a YOLOv8 object detection model trained to identify various waste categories. Adjust the confidence threshold to see more or fewer detections.")
'''

with open("app.py", "w") as f:
    f.write(streamlit_code)

print("Streamlit app.py saved!")
