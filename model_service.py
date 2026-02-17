import tensorflow as tf
import numpy as np

def create_model():
    # This must EXACTLY match the architecture shown in your client.py logs
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(230,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2), # Standard dropout rate
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='softmax') # Assuming 100 classes based on your log
    ])
    return model

# 1. Build the skeleton
MODEL = create_model()

# 2. Load only the weights
# Make sure the file name matches exactly what the server saved
MODEL.load_weights("trained_model.h5")

def predict_disease(features):
    # Reshape features for the model (1 sample, 230 features)
    input_data = np.array(features).reshape(1, 230)
    # Get probabilities for all classes
    prediction = MODEL.predict(input_data)
    return prediction