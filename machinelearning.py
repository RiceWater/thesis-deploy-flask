from PIL import Image
from io import BytesIO
from flask import Flask, jsonify


from tensorflow import keras
from keras.models import Model
import tensorflow as tf
import numpy as np

INPUT_SHAPE = (256,256)

# model = tf.keras.models.load_model('model\PCNN0206_01B.keras') # This is Final-Model_4-0-1.keras
# rnr_model = tf.keras.models.load_model('model\\RNR\\Old\\Model_V4-0-1.keras') # This is used since it performed slightly better during actual deployment
# model = tf.keras.models.load_model('model\Experimental\Final-B4_1_3-converted.keras') #This is the binary model sir Josh suggested
rnr_model = tf.keras.models.load_model('model\\RNR\\Final-B4_0_1-converted.keras')
ini_model = tf.keras.models.load_model('model\\INI\\Final-Adr-INI-5_1_1-converted.keras')

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded)).convert('RGB')
    return pil_image

def preprocess_image(image: Image.Image):
    resize = tf.image.resize(image, INPUT_SHAPE)
    ready_image = np.expand_dims(resize/255, 0)
    return ready_image

def predict_image(image: np.ndarray):
    y_rnr = rnr_model.predict(image)
    y_rnr_max = np.argmax(y_rnr)
    img_class = "Road" if y_rnr_max == 1 else "Not Road"

    if img_class != "Road":
        response = {"class" : img_class}
        return jsonify(response)
    
    y_ini = ini_model.predict(image)
    y_ini_max = np.argmax(y_ini)
    ini_class = "Issue" if y_ini_max == 1 else "Not Issue"
    y_ini_list = [float(y_ini[0][0]), float(y_ini[0][1])]
    ini_labels = ["Not Issue", "Issue"]
    response = {
            "class" : ini_class, # String
            "rnr_class": img_class, # String
            "accuracy":  y_ini_list, # List of floats           If failed, convert to string
            "ini_labels": ini_labels, # List of strings         If failed, convert to string
            }
    
    return jsonify(response)

# # FUNCTION FOR:
# # Final-B4_1_3-converted.keras
# def predict_image(image: np.ndarray):
#     y = model.predict(image)
#     y = np.argmax(y)    # Returns 0 or 1
#     print(y)
#     img_class = "Road" if y == 1 else "Not Road"
#     return img_class

# # FUNCTION FOR   4-ClASS   MODEL
# # Model_V4-0-1.keras
# # PCNN0206_01B.keras
# def predict_image(image: np.ndarray):
#     y = model.predict(image)
#     y = np.argmax(y)
#     img_class = "Road" if y == 2 else "Not Road"
#     return img_class


