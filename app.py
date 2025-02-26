import os
# Option 1: Enable GPU memory growth
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU(s).")
    except RuntimeError as e:
        print("Error enabling memory growth:", e)

# Option 2: Alternatively, disable GPU usage (uncomment the following two lines if needed)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# print("Running on CPU only.")

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
from utils.ocr_utils import (
    load_roboflow_project, load_yolo_model, train_yolo_model,
    run_yolo_inference, visualize_detections, extract_text_with_pytesseract,
    extract_text_with_trocr, preprocess_image_for_ocr, parse_clinical_data
)
from utils.model_utils import preprocess_data_for_glaucoma, predict_glaucoma

st.title("Glaucoma Detection from Clinical Report Data")
# ... rest of your code follows

# --- Option 2: Force CPU Usage (Uncomment these lines if needed) ---
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# print("Running on CPU only.")



import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tempfile

from utils.ocr_utils import (
    load_roboflow_project, load_yolo_model, train_yolo_model,
    run_yolo_inference, visualize_detections, extract_text_with_pytesseract,
    extract_text_with_trocr, preprocess_image_for_ocr, parse_clinical_data
)
from utils.model_utils import preprocess_data_for_glaucoma, predict_glaucoma

st.title("Glaucoma Detection from Clinical Report Data")

# Sidebar Navigation
app_mode = st.sidebar.selectbox(
    "Select App Mode",
    ["OCR Extraction", "Glaucoma Prediction", "Train YOLOv8", "Explainability"]
)

# --- Mode 1: OCR Extraction ---
if app_mode == "OCR Extraction":
    st.header("Extract Clinical Data from Medical Report")
    image_file = st.file_uploader("Upload Medical Report Image", type=["png", "jpg", "jpeg"])
    ocr_method = st.selectbox("Select OCR Method", ["PyTesseract", "TrOCR"])
    
    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Perform OCR based on selected method
        if ocr_method == "PyTesseract":
            ocr_text = extract_text_with_pytesseract(image)
        else:
            ocr_text = extract_text_with_trocr(image)
            
        st.text_area("Extracted Text", ocr_text, height=200)
        
        # Parse OCR text into structured clinical data
        clinical_data_df = parse_clinical_data(ocr_text)
        st.dataframe(clinical_data_df)
        
        # Option to download the extracted data as CSV (if desired)
        csv = clinical_data_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "clinical_data.csv", "text/csv")

# --- Mode 2: Glaucoma Prediction ---
elif app_mode == "Glaucoma Prediction":
    st.header("Glaucoma Prediction from Clinical Data")
    
    # --- Clinical Data Section ---
    data_source = st.radio("Select Clinical Data Source", ["Upload CSV", "Paste CSV Data"])
    
    if data_source == "Upload CSV":
        csv_file = st.file_uploader("Upload CSV File", type=["csv"])
        if csv_file:
            clinical_data_df = pd.read_csv(csv_file)
            st.dataframe(clinical_data_df)
    else:
        csv_text = st.text_area("Paste CSV Data Here", "")
        if csv_text:
            try:
                clinical_data_df = pd.read_csv(pd.compat.StringIO(csv_text))
                st.dataframe(clinical_data_df)
            except Exception as e:
                st.error("Error parsing CSV data. Please check the format.")
    
    # --- Model Loading Section ---
    model_method = st.radio("Select Model Loading Method", ["Upload Model File", "Use Local Path"])
    
    if model_method == "Upload Model File":
        model_file = st.file_uploader("Upload Glaucoma Detection Model (.h5)", type=["h5"])
        if model_file and 'clinical_data_df' in locals():
            # Save the uploaded model file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                tmp.write(model_file.getvalue())
                tmp_path = tmp.name
            model = load_model(tmp_path)
            processed_data = preprocess_data_for_glaucoma(clinical_data_df)
            predicted_class, confidence = predict_glaucoma(model, processed_data)
            st.write(f"**Predicted Class:** {predicted_class} | **Confidence:** {confidence:.2f}")
    elif model_method == "Use Local Path":
        model_path = st.text_input("Enter path to the Glaucoma Detection Model (.h5)", "models/glaucoma_model.h5")
        if model_path and 'clinical_data_df' in locals():
            try:
                model = load_model(model_path)
                processed_data = preprocess_data_for_glaucoma(clinical_data_df)
                predicted_class, confidence = predict_glaucoma(model, processed_data)
                st.write(f"**Predicted Class:** {predicted_class} | **Confidence:** {confidence:.2f}")
            except Exception as e:
                st.error(f"Error loading model from {model_path}: {e}")

# --- Mode 3: Train YOLOv8 ---
elif app_mode == "Train YOLOv8":
    st.header("Train YOLOv8 for OCR Enhancement")
    api_key = st.text_input("Enter Roboflow API Key", value="z8rBmkIxjqIa2aAm53sw")
    project_name = st.text_input("Enter Roboflow Project Name", value="glaucoma-detection")
    version_number = st.number_input("Enter Dataset Version", min_value=1, value=5)
    epochs = st.number_input("Number of Epochs", min_value=1, max_value=200, value=50)
    imgsz = st.number_input("Image Size", min_value=320, max_value=1280, value=640)
    batch = st.number_input("Batch Size", min_value=1, max_value=64, value=16)
    
    if st.button("Train YOLOv8"):
        dataset = load_roboflow_project(api_key, project_name, version_number)
        model = load_yolo_model("yolov8n.pt")
        train_yolo_model(model, dataset.location, epochs=epochs, imgsz=imgsz, batch=batch)
        st.success("✅ YOLOv8 Training Completed")

# --- Mode 4: Explainability ---
elif app_mode == "Explainability":
    st.header("Model Explainability (Grad-CAM)")
    st.info("This section is primarily for image-based explainability and may not be used for tabular data models.")
    # (Optional: Implement Grad-CAM visualization for image models here.)
