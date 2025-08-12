import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.layers import InputLayer, BatchNormalization, Dropout, Attention
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_model(input_path: str, output_path: str):
    """Enhanced model converter with architecture matching"""
    logger.info(f"Converting {input_path} to compatible format...")

    try:
        # First try direct load
        try:
            model = load_model(input_path)
            logger.info("Model loaded directly")
        except Exception as e:
            logger.warning(f"Direct load failed: {str(e)}")
            logger.info("Building architecture-matched model...")

            # Create model matching original architecture
            model = tf.keras.Sequential([
                InputLayer(input_shape=(128, 128, 1), name='input_1'),

                # Block 1
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d_1'),
                BatchNormalization(name='batch_normalization_1'),
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d_2'),
                BatchNormalization(name='batch_normalization_2'),
                tf.keras.layers.MaxPooling2D((2,2), name='max_pooling2d_1'),
                Dropout(0.25, name='dropout_1'),

                # Block 2
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_3'),
                BatchNormalization(name='batch_normalization_3'),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_4'),
                BatchNormalization(name='batch_normalization_4'),
                tf.keras.layers.MaxPooling2D((2,2), name='max_pooling2d_2'),
                Dropout(0.25, name='dropout_2'),

                # Block 3
                tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv2d_5'),
                BatchNormalization(name='batch_normalization_5'),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv2d_6'),
                BatchNormalization(name='batch_normalization_6'),
                tf.keras.layers.MaxPooling2D((2,2), name='max_pooling2d_3'),
                Dropout(0.25, name='dropout_3'),

                # Classifier
                tf.keras.layers.Flatten(name='flatten_1'),
                tf.keras.layers.Dense(512, activation='relu', name='dense_1'),
                BatchNormalization(name='batch_normalization_7'),
                Dropout(0.5, name='dropout_4'),
                tf.keras.layers.Dense(36, activation='softmax', name='dense_2')
            ])

            # Load weights
            model.load_weights(input_path, by_name=True, skip_mismatch=True)
            logger.info("Weights loaded with architecture matching")

        # Save in compatible format
        save_model(model, output_path, save_format='h5')
        logger.info(f"Successfully saved converted model to {output_path}")

        return True

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise

if __name__ == "__main__":
    convert_model(
        input_path='assets/models/final_model.h5',
        output_path='assets/models/converted_model.h5'
    )
