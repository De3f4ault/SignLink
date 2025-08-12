import cv2
import threading
import time
from queue import Queue
import numpy as np
import logging

class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.logger = logging.getLogger(__name__)
        try:
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open video source")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Warmup camera
            for _ in range(5):
                self.cap.read()

            self.frame_queue = Queue(maxsize=2)
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            self.logger.info("Camera stream initialized successfully")
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {str(e)}")
            raise

    def _update(self):
        while not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    continue

                if self.frame_queue.full():
                    self.frame_queue.get()  # Discard old frame
                self.frame_queue.put(frame)
            except Exception as e:
                self.logger.error(f"Error in camera update thread: {str(e)}")
                break

    def read(self):
        """Get latest frame with error handling"""
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get()
            return None
        except Exception as e:
            self.logger.error(f"Error reading frame: {str(e)}")
            return None

    def get_roi(self, size=300):
        """Get square region of interest from center of frame"""
        frame = self.read()
        if frame is None:
            return None

        height, width = frame.shape[:2]
        x1 = (width - size) // 2
        y1 = (height - size) // 2
        return frame[y1:y1+size, x1:x1+size]

    def release(self):
        """Cleanup resources"""
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
        self.logger.info("Camera resources released")
