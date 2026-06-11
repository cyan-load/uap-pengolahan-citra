import os

from preprocess import prepare_batch


TEST_IMAGE = "dataset/train/recycle"


def get_first_image(folder):
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(folder, file)


image_path = get_first_image(TEST_IMAGE)

cnn_img, glcm_img = prepare_batch(image_path)

print("Path:", image_path)
print("CNN Shape :", cnn_img.shape)
print("GLCM Shape:", glcm_img.shape)