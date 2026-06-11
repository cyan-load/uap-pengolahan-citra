import cv2
import numpy as np


IMG_SIZE = (224, 224)


def load_image(image_path):
    """
    Membaca gambar dari path.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    return image


def preprocess_for_cnn(image_path):
    """
    Preprocessing untuk CNN (MobileNetV2).

    Output:
        numpy array shape (224,224,3)
    """

    image = load_image(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMG_SIZE)

    image = image.astype(np.float32) / 255.0

    return image


def preprocess_for_glcm(image_path):
    """
    Preprocessing untuk GLCM.

    Output:
        grayscale image 224x224
    """

    image = load_image(image_path)

    image = cv2.resize(image, IMG_SIZE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


def prepare_batch(image_path):
    """
    Menghasilkan data untuk CNN dan GLCM sekaligus.
    """

    cnn_image = preprocess_for_cnn(image_path)

    glcm_image = preprocess_for_glcm(image_path)

    return cnn_image, glcm_image


if __name__ == "__main__":

    sample_path = "dataset/train/recycle"

    print("Preprocessing module berhasil dimuat.")