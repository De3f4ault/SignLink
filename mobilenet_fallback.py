import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

def build_mobilenet():
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Adjust for your classes
    ])
    # Quantize model (reduces size by 4x)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quantized_model = converter.convert()
    return quantized_model
