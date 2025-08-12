import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class CameraFeed(ttk.Frame):
    def __init__(self, parent, camera, detector):
        super().__init__(parent)
        self.camera = camera
        self.detector = detector
        self.video_label = ttk.Label(self)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def update_frame(self, frame):
        try:
            # Detect hands and draw on frame
            roi = self.detector.detect(frame)

            # Convert and resize frame
            frame = cv2.resize(frame, (640, 480))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            return roi  # Return ROI for prediction
        except Exception as e:
            print(f"Frame update error: {str(e)}")
            return None
