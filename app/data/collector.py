import os
import cv2
import time
from datetime import datetime
from ..core.processor import preprocess_frame

class DataCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        self.camera = cv2.VideoCapture(0)
        self.running = False

    def collect_samples(self, class_label, samples=1000, delay=0.1):
        """Collect samples for a given class"""
        class_dir = os.path.join(self.output_dir, class_label)
        os.makedirs(class_dir, exist_ok=True)

        self.running = True
        count = 0

        while self.running and count < samples:
            ret, frame = self.camera.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            roi = self._get_roi(frame)
            processed = preprocess_frame(roi)

            # Display
            cv2.imshow("Data Collection", frame)
            cv2.imshow("Processed", processed)

            # Save every nth frame
            if count % 5 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(class_dir, f"{timestamp}.jpg")
                cv2.imwrite(save_path, roi)
                print(f"Saved sample {count}/{samples} to {save_path}")

            count += 1
            time.sleep(delay)

            if cv2.waitKey(1) == 27:  # ESC key
                self.running = False

        cv2.destroyAllWindows()

    def _get_roi(self, frame, size=300):
        """Get square region of interest"""
        height, width = frame.shape[:2]
        x1 = (width - size) // 2
        y1 = (height - size) // 2
        return frame[y1:y1+size, x1:x1+size]
