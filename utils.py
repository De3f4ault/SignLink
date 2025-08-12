import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.utils import to_categorical

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Sign language vocabulary
WORDS = ["hello", "thanks", "iloveyou", "yes", "no", "please", "sorry", "help", "more", "finished"]
NUM_CLASSES = len(WORDS)

# Make available for import
words = WORDS
sentences = []

# Sequence configuration
SEQUENCE_LENGTH = 30  # Number of frames in a sequence
FRAME_DIMENSION = 126  # 21 landmarks * (x,y,z) coordinates

def extract_keypoints(results):
    """Extracts and normalizes hand landmarks from MediaPipe results"""
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])

    # Pad with zeros if no hands detected
    while len(keypoints) < FRAME_DIMENSION:
        keypoints.append(0.0)

    keypoints = np.array(keypoints[:FRAME_DIMENSION])

    # Normalize keypoints
    if np.any(keypoints):
        keypoints = (keypoints - np.mean(keypoints)) / np.std(keypoints)

    return keypoints

def draw_landmarks(image, results):
    """Draws hand landmarks and connections on the image"""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
    return image

def load_dataset(data_path):
    """Loads training data from numpy arrays"""
    X = []
    y = []
    label_map = {label:num for num, label in enumerate(WORDS)}

    for word in os.listdir(data_path):
        word_path = os.path.join(data_path, word)
        if not os.path.isdir(word_path):
            continue

        for seq_file in os.listdir(word_path):
            if seq_file.endswith('.npy'):
                sequence = np.load(os.path.join(word_path, seq_file))
                X.append(sequence)
                y.append(label_map[word])

    X = np.array(X)
    y = to_categorical(y, num_classes=NUM_CLASSES)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_frame(frame):
    """Preprocesses a frame for prediction"""
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def initialize_hands_model():
    """Initializes MediaPipe Hands model with optimized settings"""
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
        model_complexity=1
    )
