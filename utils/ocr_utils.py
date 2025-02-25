import numpy as np
import cv2
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from roboflow import Roboflow
from ultralytics import YOLO

# ----- OCR Functions -----
def extract_text_with_trocr(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    inputs = processor(image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(inputs)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def extract_text_with_pytesseract(image):
    return pytesseract.image_to_string(image, lang='eng')

def preprocess_image_for_ocr(image):
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_img = Image.fromarray(thresh)
    plt.imshow(preprocessed_img, cmap='gray')
    plt.title("Preprocessed Image for OCR")
    plt.axis("off")
    plt.show()
    return preprocessed_img

def parse_clinical_data(ocr_text):
    """
    Parse the extracted OCR text to structured clinical data.
    This placeholder uses regex to extract sample parameters such as:
      - Age (e.g., "Age: 65")
      - Intraocular Pressure (e.g., "Intraocular Pressure: 22")
      - Cup-to-Disc Ratio (e.g., "Cup-to-Disc Ratio: 0.6")
    Adjust the regex patterns based on your report format.
    """
    data = {}

    age_match = re.search(r"Age[:\s]+(\d+)", ocr_text, re.IGNORECASE)
    data["patient_age"] = float(age_match.group(1)) if age_match else 0.0

    iop_match = re.search(r"Intraocular Pressure[:\s]+(\d+\.?\d*)", ocr_text, re.IGNORECASE)
    data["intraocular_pressure"] = float(iop_match.group(1)) if iop_match else 0.0

    cdr_match = re.search(r"Cup[-\s]?to[-\s]?Disc Ratio[:\s]+(\d+\.?\d*)", ocr_text, re.IGNORECASE)
    data["cup_to_disc_ratio"] = float(cdr_match.group(1)) if cdr_match else 0.0

    # Add more parameters as needed
    return pd.DataFrame([data])

# ----- Roboflow / YOLO Helper Functions -----
def load_roboflow_project(api_key, project_name, version_number):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_name)
    dataset = project.version(str(version_number)).download("yolov8")
    return dataset

def load_yolo_model(weights_path):
    return YOLO(weights_path)

def train_yolo_model(model, dataset_path, epochs=50, imgsz=640, batch=16):
    model.train(
        data=f"{dataset_path}/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch
    )
    print("✅ YOLOv8 Training Completed")

def run_yolo_inference(model, image, conf=0.25, save=False):
    # Convert PIL image to numpy array if needed
    image_np = np.array(image)
    results = model.predict(source=image_np, conf=conf, save=save)
    return results

def visualize_detections(results):
    from PIL import Image
    import os
    for idx, r in enumerate(results):
        if hasattr(r, 'path') and r.path and os.path.exists(r.path):
            im = Image.open(r.path)
        elif hasattr(r, 'orig_img'):
            im = Image.fromarray(r.orig_img)
        else:
            print(f"⚠️ Unable to load image for result {idx}")
            continue

        plt.figure(figsize=(10, 8))
        plt.imshow(im)
        plt.title("Detected Text Regions")
        plt.axis("off")
        plt.show()
