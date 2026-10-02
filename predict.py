import tensorflow as tf
import numpy as np
from PIL import Image
import sys

# LOAD YOUR SAVED MODEL
model = tf.keras.models.load_model('Daud_model.keras')

# CLASS NAMES
class_names = ['animals', 'crops', 'flower']

# LOAD AND PREDICT IMAGE
def predict_image(image_path):
    # open and resize image to 64x64 (same size we trained on)
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img = np.array(img) / 255.0        # normalize
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # PREDICT
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"All scores: animals={predictions[0][0]*100:.1f}% crops={predictions[0][1]*100:.1f}% flower={predictions[0][2]*100:.1f}%")

# RUN
predict_image(sys.argv[1])





#py -3.11 predict.py test.jpg