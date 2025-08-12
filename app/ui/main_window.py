import tkinter as tk
from tkinter import ttk
import logging

from ..core.camera import CameraStream
from ..core.hand_detector import HandDetector
from ..core.model import SignLanguageModel
from ..core.language_processor import EnhancedLanguageProcessor
from .components.camera_feed import CameraFeed
from .components.prediction_display import PredictionDisplay


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SignLink - Real-Time ASL Recognition")
        self.geometry("1200x800")
        self.configure_logging()
        self.logger = logging.getLogger(__name__)

        try:
            # Initialize core components
            self.logger.info("Initializing camera...")
            self.camera = CameraStream()

            self.logger.info("Initializing hand detector...")
            self.hand_detector = HandDetector()  # ✅ Create detector instance

            self.logger.info("Loading model...")
            self.model = SignLanguageModel(
                model_path='assets/models/converted_model.h5',
                input_shape=(128, 128, 1),
                num_classes=36
            )

            self.logger.info("Initializing language processor...")
            self.language_processor = EnhancedLanguageProcessor()

            # Setup UI - ✅ pass hand_detector to CameraFeed
            self.setup_ui()

            # Start processing loop
            self.after(100, self.process_frame)

        except Exception as e:
            logging.critical(f"Application initialization failed: {str(e)}", exc_info=True)
            self.destroy()
            raise

    def configure_logging(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('signlink.log'),
                logging.StreamHandler()
            ]
        )

    def setup_ui(self):
        """Initialize all UI components"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Camera Feed (Left 60%)
        cam_frame = ttk.Frame(main_frame)
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ✅ Pass hand_detector to CameraFeed
        self.camera_feed = CameraFeed(cam_frame, self.camera, self.hand_detector)
        self.camera_feed.pack(fill=tk.BOTH, expand=True)

        # Prediction Panel (Right 40%)
        pred_frame = ttk.Frame(main_frame, width=400)
        pred_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.prediction_display = PredictionDisplay(pred_frame, self.language_processor)
        self.prediction_display.pack(fill=tk.BOTH, expand=True)

    def process_frame(self):
        """Main processing loop with error handling"""
        try:
            frame = self.camera.read()
            if frame is not None:
                # Get ROI from camera feed (hand detection happens here)
                roi = self.camera_feed.update_frame(frame)

                if roi is not None:
                    # Make prediction
                    prediction = self.model.predict(roi)
                    if prediction:
                        # Update display
                        self.prediction_display.update_prediction(prediction)

                        # Optionally update language processor if using character-level prediction
                        if self.language_processor and 'class' in prediction:
                            self.language_processor.update_current_word(prediction['class'])

            # Schedule next frame
            self.after(30, self.process_frame)

        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
            self.after(30, self.process_frame)

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'camera'):
            self.camera.release()
