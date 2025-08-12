from core.custom_cnn import build_cnn
from utils.preprocess import load_data

X_train, y_train = load_data('dataset/')  # Implement your loader
model = build_cnn()
model.fit(X_train, y_train, epochs=20, batch_size=32)
model.save('custom_cnn.keras')
