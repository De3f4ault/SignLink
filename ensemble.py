import numpy as np

class EnsemblePredictor:
    def __init__(self, cnn_model, mobilenet_model):
        self.cnn = cnn_model
        self.mobilenet = mobilenet_model

    def predict(self, roi):
        # CNN prediction (fast)
        cnn_pred = self.cnn.predict(roi[np.newaxis, ...])
        cnn_conf = np.max(cnn_pred)

        if cnn_conf >= 0.8:  # High confidence
            return np.argmax(cnn_pred), cnn_conf

        # Fallback to MobileNet (robust)
        mobilenet_pred = self.mobilenet.predict(cv2.resize(roi, (224, 224))[np.newaxis, ...])
        mobilenet_conf = np.max(mobilenet_pred)

        # Weighted average
        combined = (cnn_conf * cnn_pred) + (mobilenet_conf * mobilenet_pred)
        return np.argmax(combined), np.max(combined)
