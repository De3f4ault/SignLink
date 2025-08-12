import cv2
import numpy as np

def preprocess_frame(frame, target_size=(128, 128)):
    """Process frame for model input"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    processed = cv2.resize(thresh, target_size)
    return np.expand_dims(processed, axis=-1)
