from ultralytics import YOLO
from google.colab import files
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load the newly trained custom YOLO model
model_v2 = YOLO("/content/drive/MyDrive/waste_project/improved_model/waste_final_v2_best.pt")

print("Model loaded successfully!")
if hasattr(model_v2, 'names'):
    print(f"Model recognizes the following classes: {list(model_v2.names.values())}")
else:
    print("Could not retrieve class names from the model.")

print("\nUpload an image file for testing with the new model:")
uploaded_new = files.upload()

for filename_new in uploaded_new.keys():
    print(f"\nProcessing uploaded file: {filename_new}")
    # Convert uploaded file to PIL Image for consistent handling and to display original if no detections
    img_bytes = uploaded_new[filename_new]
    uploaded_image_pil = Image.open(io.BytesIO(img_bytes))

    # Run inference with a very low confidence threshold for debugging
    # This helps determine if *any* detections are made, no matter how weak.
    debug_conf_threshold = 0.01
    print(f"Attempting detection with confidence threshold: {debug_conf_threshold}")
    results_v2 = model_v2.predict(source=uploaded_image_pil, conf=debug_conf_threshold, save=False, verbose=False)

    # Plot the results and display using matplotlib
    if results_v2 and len(results_v2[0].boxes) > 0:
        print("Objects detected! Displaying annotated image with new model.")
        plotted_image_v2 = results_v2[0].plot() # This returns a numpy array with the annotated image
        plt.figure(figsize=(10, 10))
        plt.imshow(plotted_image_v2[:, :, ::-1]) # OpenCV uses BGR, Matplotlib uses RGB
        plt.axis("off")
        plt.title(f"Detection Results for {filename_new} (New Model) - Conf: {debug_conf_threshold}")
        plt.show()

        # Print detected classes and confidence scores
        print("\n--- Detected Objects ---")
        for r in results_v2:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # Get class name from model's names attribute
                class_name = model_v2.names[cls] if hasattr(model_v2, 'names') and cls < len(model_v2.names) else f"Class {cls}"
                print(f"Detected: {class_name} with confidence {conf:.2f}")
        print("------------------------")

    else:
        print(f"No objects detected in {filename_new} even with a very low confidence threshold ({debug_conf_threshold}).")
        print("Displaying original uploaded image for verification.")
        plt.figure(figsize=(10, 10))
        plt.imshow(uploaded_image_pil)
        plt.axis("off")
        plt.title(f"Original Uploaded Image: {filename_new} (No Detections)")
        plt.show()
