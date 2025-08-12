import cv2
import numpy as np

def preprocess_frame(frame, target_size=(128, 128)):
    """Enhanced preprocessing pipeline with adaptive thresholding"""
    if frame is None or frame.size == 0:
        return None

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur (reduced kernel size from 5x5 to 3x3 for sharper features)
        blur = cv2.GaussianBlur(gray, (3,3), 2)

        # Adaptive thresholding with Gaussian
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Optional: OTSU thresholding (can be commented out if adaptive works well)
        _, otsu = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Resize and expand dimensions for model input
        processed = cv2.resize(otsu, target_size)
        return np.expand_dims(processed, axis=-1)

    except Exception as e:
        logging.error(f"Image processing error: {str(e)}")
        return None
