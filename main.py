import cv2
import mediapipe as mp
import numpy as np
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils import extract_keypoints, words, sentences
from model import load_model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load the trained model - UPDATE THIS PATH
MODEL_PATH = 'action.h5'  # or 'custom_cnn.keras'

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Critical error loading model: {e}")
    exit(1)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Prediction variables
sequence = []
predictions = []
threshold = 0.8
current_word = ""
last_prediction_time = 0
prediction_interval = 0.5  # seconds

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Mirror and process frame
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    # Draw landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep last 30 frames

        # Make predictions at intervals
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if len(sequence) == 30 and current_time - last_prediction_time > prediction_interval:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            if res[np.argmax(res)] > threshold:
                current_word = words[np.argmax(res)]
                print(f"Prediction: {current_word} (Confidence: {res[np.argmax(res)]:.2f})")

                if not sentences or sentences[-1] != current_word:
                    sentences.append(current_word)

            last_prediction_time = current_time

    # Display info
    cv2.putText(frame, f"Current: {current_word}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Sentence: {' '.join(sentences)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Sign Language Recognition', frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        current_word = ""
    elif key == ord('a'):
        sentences.clear()

cap.release()
cv2.destroyAllWindows()
