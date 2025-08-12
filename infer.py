import cv2
import tensorflow as tf
from core.hand_detector import HandDetector
from core.ensemble import EnsemblePredictor
from core.language_processor import LanguageProcessor

def main():
    # Initialize models first
    detector = HandDetector()
    try:
        cnn = tf.keras.models.load_model('custom_cnn.h5')
        mobilenet = tf.lite.Interpreter('quant_mobilenet.tflite')
        predictor = EnsemblePredictor(cnn, mobilenet)
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return

    # Initialize language processor
    try:
        lp = LanguageProcessor("utils/words.txt")
    except Exception as e:
        print(f"Error initializing language processor: {str(e)}")
        lp = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        try:
            roi = detector.detect(frame)
            if roi is not None:
                # Get predicted sign (class ID or character)
                pred_class, conf = predictor.predict(roi)
                pred_char = chr(ord('a') + pred_class)  # Convert class to letter

                # Update word/sentence if language processor is available
                if lp:
                    suggestions = lp.update_current_word(pred_char)
                    print(f"Current word: {''.join(lp.current_word)} | Suggestions: {suggestions}")

                    # Display sentence
                    cv2.putText(frame, f"Sentence: {' '.join(lp.sentence)}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display prediction
                cv2.putText(frame, f"Sign: {pred_char} ({conf:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("SignLink", frame)
            key = cv2.waitKey(1)
            if key == ord(' '):  # Spacebar to finalize word
                if lp:
                    lp.finalize_word()
            elif key == ord('q'):
                break

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
