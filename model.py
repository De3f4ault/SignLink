import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as tf_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np

# Vocabulary of signs to recognize
WORDS = ["hello", "thanks", "iloveyou", "yes", "no", "please", "sorry", "help", "more", "finished"]
NUM_CLASSES = len(WORDS)

def create_model():
    """Creates a new LSTM model for sign language recognition"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(30, 126)),
        Dropout(0.3),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def load_model(model_path):
    """Loads a pre-trained model with GPU optimization"""
    # Configure GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU config error: {e}")

    try:
        model = tf_load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a new model instead...")
        model = create_model()
        return model
