import os
import cv2
import pickle
import numpy as np
import tensorflow as tf

from skimage.feature import graycomatrix
from skimage.feature import graycoprops

# ==========================
# PATH
# ==========================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "models",
    "model_hybrid.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "models",
    "class_names.pkl"
)

# ==========================
# LOAD MODEL
# ==========================

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(CLASS_PATH, "rb") as f:
    class_names = pickle.load(f)

IMG_SIZE = (224, 224)

# ==========================
# GLCM
# ==========================

def extract_glcm(gray_img):

    glcm = graycomatrix(
        gray_img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = float(
        graycoprops(glcm, "contrast")[0, 0]
    )

    correlation = float(
        graycoprops(glcm, "correlation")[0, 0]
    )

    energy = float(
        graycoprops(glcm, "energy")[0, 0]
    )

    homogeneity = float(
        graycoprops(glcm, "homogeneity")[0, 0]
    )

    return {
        "contrast": contrast,
        "correlation": correlation,
        "energy": energy,
        "homogeneity": homogeneity
    }

# ==========================
# PREDICT
# ==========================

def predict_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    # CNN INPUT

    cnn_input = image.astype(
        np.float32
    ) / 255.0

    cnn_input = np.expand_dims(
        cnn_input,
        axis=0
    )

    # GLCM INPUT

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    glcm_values = extract_glcm(
        gray
    )

    glcm_input = np.array([
        [
            glcm_values["contrast"],
            glcm_values["correlation"],
            glcm_values["energy"],
            glcm_values["homogeneity"]
        ]
    ])

    # PREDICT

    prediction = model.predict(
        [cnn_input, glcm_input],
        verbose=0
    )

    idx = np.argmax(
        prediction
    )

    confidence = float(
        prediction[0][idx] * 100
    )

    label = class_names[idx]

    return {
        "label": label,
        "confidence": confidence,
        "glcm": glcm_values,
        "probabilities": {
            class_names[i]: float(prediction[0][i] * 100)
            for i in range(len(class_names))
        }
    }