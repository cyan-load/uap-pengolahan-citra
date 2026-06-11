import os
import sys
import numpy as np
import pandas as pd

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# import preprocessing
sys.path.append(os.path.abspath("src/preprocessing"))

from preprocess import preprocess_for_cnn


OUTPUT_DIR = "outputs/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# MobileNetV2 sebagai feature extractor
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

feature_extractor = Model(
    inputs=base_model.input,
    outputs=base_model.output
)

def extract_cnn_features(image_path):

    img = preprocess_for_cnn(image_path)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    features = feature_extractor.predict(
        img,
        verbose=0
    )

    return features.flatten()

def process_dataset(split_name):

    dataset_path = os.path.join(
        "dataset",
        split_name
    )

    rows = []

    for label in os.listdir(dataset_path):

        label_path = os.path.join(
            dataset_path,
            label
        )

        if not os.path.isdir(label_path):
            continue

        print(
            f"[INFO] Processing {split_name}/{label}"
        )

        for image_name in os.listdir(label_path):

            if not image_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                label_path,
                image_name
            )

            try:

                features = extract_cnn_features(
                    image_path
                )

                row = {
                    "image_name": image_name,
                    "split": split_name,
                    "label": label
                }

                for i, value in enumerate(features):
                    row[f"cnn_{i}"] = value

                rows.append(row)

            except Exception as e:

                print(
                    f"[ERROR] {image_name}: {e}"
                )

    return rows

def main():

    all_rows = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        rows = process_dataset(split)

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    save_path = os.path.join(
        OUTPUT_DIR,
        "cnn_features.csv"
    )

    df.to_csv(
        save_path,
        index=False
    )

    print("\n===================")
    print("CNN FEATURE DONE")
    print("===================")

    print(df.shape)

    print(
        f"Saved : {save_path}"
    )


if __name__ == "__main__":
    main()