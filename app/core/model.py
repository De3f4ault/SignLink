import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model as tf_load_model
import numpy as np

# Vocabulary of signs to recognize (should match your training data)
WORDS = ["hello", "thanks", "iloveyou", "yes", "no", "please", "sorry", "help", "more", "finished"]
NUM_WORDS = len(WORDS)

def create_model():
    """
    Creates and compiles a new LSTM model with optimized architecture
    for sign language recognition.
    """
    model = Sequential([
        # First LSTM layer with return sequences
        LSTM(128, return_sequences=True, input_shape=(30, 126),
             kernel_initializer='he_normal', recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM layer
        LSTM(256, return_sequences=True,
             kernel_initializer='he_normal', recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.3),

        # Third LSTM layer
        LSTM(128, return_sequences=False,
             kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),

        # Dense layers
        Dense(256, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.4),

        # Output layer
        Dense(NUM_WORDS, activation='softmax')
    ])

    # Custom optimizer configuration
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print("New model created successfully.")
    return model

def load_model(model_path):
    """
    Loads a pre-trained model with GPU optimization and fallback to CPU.
    If loading fails, creates a new model.
    """
    # Configure GPU for better performance
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
             print(f"GPU config error: {e}")

    try:
        model = tf_load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def create_model():
    """Model creation function if needed"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(30, 126)),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(len(words), activation='softmax')
    ])

    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model



def get_callbacks():
    """
    Returns a list of callbacks for model training
    """
    return [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
    ]

def train_model(X_train, y_train, X_val, y_val, epochs=100):
    """
    Complete training pipeline
    """
    model = create_model()
    callbacks = get_callbacks()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model, history

if __name__ == "__main__":
    # Example usage for training
    print("Initializing model...")
    model = create_model()
    model.summary()
