import numpy as np
import pandas as pd
import tensorflow as tf

def build_glaucoma_model(input_dim=8):
    """
    Build a simple dense model for glaucoma detection based on tabular data.
    Assumes input features:
      - Min_GCL_IPL_OD
      - Min_GCL_IPL_OS
      - Avg_GCL_IPL_OD
      - Avg_GCL_IPL_OS
      - Vertical_CDR_OD
      - Vertical_CDR_OS
      - Rim_Area_OD
      - Rim_Area_OS
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_data_for_glaucoma(df):
    """
    Convert the clinical data DataFrame into a NumPy array suitable for the glaucoma model.
    Expects the following features:
      - Min_GCL_IPL_OD
      - Min_GCL_IPL_OS
      - Avg_GCL_IPL_OD
      - Avg_GCL_IPL_OS
      - Vertical_CDR_OD
      - Vertical_CDR_OS
      - Rim_Area_OD
      - Rim_Area_OS
    If any column is missing, it is filled with 0.0.
    """
    features = [
        'Min_GCL_IPL_OD',
        'Min_GCL_IPL_OS',
        'Avg_GCL_IPL_OD',
        'Avg_GCL_IPL_OS',
        'Vertical_CDR_OD',
        'Vertical_CDR_OS',
        'Rim_Area_OD',
        'Rim_Area_OS'
    ]
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    return df[features].values.astype('float32')

def predict_glaucoma(model, processed_data):
    """
    Run prediction on the processed data and return the predicted class and confidence.
    """
    predictions = model.predict(processed_data)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return predicted_class, confidence
