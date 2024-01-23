from PIL import Image
from io import BytesIO

from tensorflow import keras
from keras.models import Model
import tensorflow as tf
import numpy as np

INPUT_SHAPE = (256,256)

D_MODEL_P1 = tf.keras.models.load_model('models\STD_V3_Part1.keras')
D_MODEL_P2 = tf.keras.models.load_model('models\STD_V3_Part2.keras')
# combined_model = Model(inputs=D_MODEL_P1.input, outputs=D_MODEL_P2(D_MODEL_P1.output))

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded)).convert('RGB')
    return pil_image

def preprocess_image(image: Image.Image):
    resize = tf.image.resize(image, INPUT_SHAPE)
    ready_image = np.expand_dims(resize/255, 0)
    return ready_image

def predict_image(image: np.ndarray):
    p2 = D_MODEL_P1.predict(image)
    y = D_MODEL_P2.predict(p2)
    y = np.argmax(y)
    img_class = "Road" if y == 3 else "Not Road"
    return img_class